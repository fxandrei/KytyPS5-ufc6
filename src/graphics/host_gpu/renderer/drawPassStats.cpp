#include "graphics/host_gpu/renderer/drawPassStats.h"

#include "common/abSwitch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// See drawPassStats.h. Output (one block per window, all lines on stdout, key=value fields; the
// text after " | " is a human-readable description):
//   draw-pass-stats: t= unix= window= frames= fps= calls= draws= ... (totals of the window)
//   draw-pass-stats-class: t= id=C<n> hash=<16 hex> draws= draws_f= share= runs_f= views_f= vp=... | ...
//   draw-pass-stats-group: t= id=G<n> class=C<n> vs= vs_hash= ps= ps_hash= draws= draws_f= share= ...
//   draw-pass-stats-cs: t= cs= hash= n= n_f= share= groups_avg= tiny_f= indirect_f=
// Class and group ids are stable within a run; the class hash only depends on formats, extents and
// depth state, so it can be compared across runs.

namespace Libs::Graphics::DrawPassStats {

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint64_t Avalanche(uint64_t x) noexcept {
	x ^= x >> 30u;
	x *= 0xbf58476d1ce4e5b9ull;
	x ^= x >> 27u;
	x *= 0x94d049bb133111ebull;
	x ^= x >> 31u;
	return x;
}

constexpr uint64_t Mix(uint64_t hash, uint64_t value) noexcept {
	return Avalanche(hash ^ (value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u)));
}

uint32_t EnvUint(const char* name, uint32_t fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') {
		return fallback;
	}
	return static_cast<uint32_t>(std::strtoul(value, nullptr, 0));
}

bool EnvFlag(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && *value != '\0' && *value != '0';
}

struct Settings {
	bool     stats       = false;
	double   period_s    = 5.0;
	uint32_t top_groups  = 25;
	uint32_t top_classes = 12;
	uint32_t top_cs      = 12;
};

Settings ReadSettings() {
	Settings settings;
	settings.stats = EnvFlag("KYTY_DRAW_PASS_STATS");
	if (const char* value = std::getenv("KYTY_DRAW_PASS_STATS_SECONDS");
	    value != nullptr && std::atof(value) > 0.0) {
		settings.period_s = std::atof(value);
	}
	settings.top_groups  = EnvUint("KYTY_DRAW_PASS_STATS_TOP", settings.top_groups);
	settings.top_classes = EnvUint("KYTY_DRAW_PASS_STATS_CLASSES", settings.top_classes);
	settings.top_cs      = EnvUint("KYTY_DRAW_PASS_STATS_CS", settings.top_cs);
	return settings;
}

const Settings& GetSettings() {
	static const Settings settings = ReadSettings();
	return settings;
}

struct ViewSize {
	uint32_t width  = 0;
	uint32_t height = 0;
	uint64_t draws  = 0;
};

// A render-target set.
struct ClassRecord {
	uint64_t              hash       = 0;
	uint32_t              id         = 0;
	bool                  depth_only = false;
	std::string           text;
	uint64_t              draws = 0; // window
	uint64_t              runs  = 0; // window: entries from a different view (see RunKey)
	uint64_t              views = 0; // window: sum over frames of the distinct views per frame
	std::vector<ViewSize> sizes;     // viewport sizes; counts are per window
	std::vector<uint64_t> frame_views;
	bool                  touched = false;
};

// A render-target set and the programs drawing into it.
struct GroupRecord {
	uint32_t id               = 0;
	uint32_t class_index      = 0;
	uint64_t vs_address       = 0;
	uint64_t vs_hash          = 0;
	uint64_t ps_address       = 0;
	uint64_t ps_hash          = 0;
	bool     ps_active        = false;
	bool     mesh             = false;
	uint64_t draws            = 0; // window counts from here
	uint64_t instances        = 0;
	uint64_t direct_draws     = 0;
	uint64_t direct_indices   = 0;
	uint64_t indirect_draws   = 0;
	uint64_t indirect_records = 0;
};

struct ComputeRecord {
	uint64_t address  = 0;
	uint64_t hash     = 0;
	uint64_t count    = 0; // window counts from here
	uint64_t groups   = 0;
	uint64_t tiny     = 0;
	uint64_t indirect = 0;
};

constexpr uint32_t FRAME_VIEWS_MAX = 64;

struct State {
	std::mutex                             mutex;
	std::unordered_map<uint64_t, uint32_t> class_index;
	std::vector<ClassRecord>               classes;
	std::unordered_map<uint64_t, uint32_t> group_index;
	std::vector<GroupRecord>               groups;
	std::unordered_map<uint64_t, uint32_t> compute_index;
	std::vector<ComputeRecord>             computes;
	std::vector<uint32_t>                  touched;
	uint64_t                               last_class_hash = 0;
	uint32_t                               last_class      = UINT32_MAX;
	uint64_t                               last_group_hash = 0;
	uint32_t                               last_group      = UINT32_MAX;
	uint64_t                               last_run_key    = 0;
	bool                                   started         = false;
	Clock::time_point                      start;
	Clock::time_point                      window_start;
	uint64_t                               frames           = 0;
	uint64_t                               draws            = 0;
	uint64_t                               depth_only_draws = 0;
	uint64_t                               indirect_draws   = 0;
	uint64_t                               indirect_records = 0;
	uint64_t                               dispatches       = 0;
};

// Never destroyed: the GPU thread can still draw while the process exits.
State& GetState() {
	static State* state = new State;
	return *state;
}

std::atomic<uint64_t> g_calls {0};

void StartIfNeeded(State& s) {
	if (!s.started) {
		s.started      = true;
		s.start        = Clock::now();
		s.window_start = s.start;
	}
}

uint32_t ToPixels(float value) {
	if (!std::isfinite(value)) {
		return 0;
	}
	return static_cast<uint32_t>(std::clamp(std::lround(std::fabs(value)), 0l, 65535l));
}

int32_t ToOffset(float value) {
	if (!std::isfinite(value)) {
		return 0;
	}
	return static_cast<int32_t>(std::clamp(std::lround(value), -65536l, 65536l));
}

struct ViewportRect {
	int32_t  x      = 0;
	int32_t  y      = 0;
	uint32_t width  = 0;
	uint32_t height = 0;
};

ViewportRect GetViewport(const CommandBuffer& buffer) {
	const auto& vp = buffer.GetRegisters().GetScreenViewport().viewports[0];
	return {ToOffset(vp.xoffset - std::fabs(vp.xscale)), ToOffset(vp.yoffset - std::fabs(vp.yscale)),
	        ToPixels(vp.xscale * 2.0f), ToPixels(vp.yscale * 2.0f)};
}

// Formats, extents and depth state only (no addresses, slices or viewports): the same in every run
// with the same output settings.
uint64_t ClassHash(const DrawTargets& targets) {
	uint64_t hash = Mix(0x6b79747970617373ull, targets.color_count);
	for (uint32_t i = 0; i < targets.color_count; i++) {
		const auto& color  = targets.colors[i];
		const auto  extent = color.Extent();
		hash               = Mix(hash, static_cast<uint64_t>(color.desc.view_info.format));
		hash = Mix(hash, (uint64_t {extent.width} << 32u) | extent.height);
		hash = Mix(hash, (uint64_t {color.target_slot} << 32u) | color.desc.info.resources.layers);
	}
	const auto& depth = targets.depth;
	if (!depth.image_id) {
		return Mix(hash, 0x6e6f6e65ull);
	}
	hash = Mix(hash, static_cast<uint64_t>(depth.desc.view_info.format));
	hash = Mix(hash, (uint64_t {depth.desc.info.extent.width} << 32u) | depth.desc.info.extent.height);
	hash = Mix(hash, depth.desc.info.resources.layers);
	return Mix(hash, (depth.depth_test_enable ? 1u : 0u) | (depth.depth_write_enable ? 2u : 0u) |
	                     (depth.stencil_test_enable ? 4u : 0u) |
	                     (static_cast<uint64_t>(depth.depth_compare_op) << 8u));
}

// A view: the targets' addresses and slices and the viewport rectangle (one shadow-map tile, one
// cascade slice, one camera...).
uint64_t RunKey(uint64_t class_hash, const DrawTargets& targets, const ViewportRect& vp) {
	uint64_t key = Mix(class_hash, (uint64_t {static_cast<uint32_t>(vp.x)} << 32u) |
	                                   static_cast<uint32_t>(vp.y));
	key          = Mix(key, (uint64_t {vp.width} << 32u) | vp.height);
	if (targets.depth.image_id) {
		key = Mix(key, targets.depth.desc.info.data.address);
		key = Mix(key, targets.depth.desc.view_info.base_layer);
	}
	if (targets.color_count != 0) {
		key = Mix(key, targets.colors[0].desc.info.data.address);
		key = Mix(key, targets.colors[0].guest_array_layer);
	}
	return key;
}

std::string LayersText(uint32_t layers) {
	return layers > 1 ? fmt::format("x{}", layers) : std::string {};
}

std::string ClassText(const DrawTargets& targets) {
	std::string text;
	if (targets.color_count == 0) {
		text = "depth-only";
	} else {
		text = fmt::format("colors={}", targets.color_count);
		for (uint32_t i = 0; i < targets.color_count; i++) {
			const auto& color  = targets.colors[i];
			const auto  extent = color.Extent();
			text += fmt::format(" [{} {} {}x{}{}]", color.target_slot,
			                    vk::to_string(color.desc.view_info.format), extent.width,
			                    extent.height, LayersText(color.desc.info.resources.layers));
		}
	}
	const auto& depth = targets.depth;
	if (!depth.image_id) {
		return text + " depth=none";
	}
	text += fmt::format(" depth=[{} {}x{}{}", vk::to_string(depth.desc.view_info.format),
	                    depth.desc.info.extent.width, depth.desc.info.extent.height,
	                    LayersText(depth.desc.info.resources.layers));
	if (!depth.depth_test_enable && !depth.depth_write_enable) {
		text += " z=off";
	} else {
		text += fmt::format(" z={}{}",
		                    depth.depth_test_enable ? vk::to_string(depth.depth_compare_op)
		                                            : std::string("Always"),
		                    depth.depth_write_enable ? "/write" : "/nowrite");
	}
	if (depth.stencil_test_enable) {
		text += " stencil";
	}
	return text + "]";
}

uint32_t FindClass(State& s, uint64_t hash, const DrawTargets& targets) {
	if (s.last_class != UINT32_MAX && s.last_class_hash == hash) {
		return s.last_class;
	}
	auto [it, inserted] = s.class_index.try_emplace(hash, static_cast<uint32_t>(s.classes.size()));
	if (inserted) {
		ClassRecord record;
		record.hash       = hash;
		record.id         = static_cast<uint32_t>(s.classes.size());
		record.depth_only = targets.color_count == 0;
		record.text       = ClassText(targets);
		s.classes.push_back(std::move(record));
	}
	s.last_class_hash = hash;
	s.last_class      = it->second;
	return it->second;
}

void CountViewSize(ClassRecord& record, const ViewportRect& vp) {
	for (auto& size: record.sizes) {
		if (size.width == vp.width && size.height == vp.height) {
			size.draws++;
			return;
		}
	}
	if (record.sizes.size() < 32) {
		record.sizes.push_back({vp.width, vp.height, 1});
	}
}

double PerFrame(uint64_t value, uint64_t frames) {
	return frames != 0 ? static_cast<double>(value) / static_cast<double>(frames) : 0.0;
}

double Percent(uint64_t value, uint64_t total) {
	return total != 0 ? 100.0 * static_cast<double>(value) / static_cast<double>(total) : 0.0;
}

void Report(State& s, Clock::time_point now) {
	const auto&  settings = GetSettings();
	const double window   = std::chrono::duration<double>(now - s.window_start).count();
	const double t        = std::chrono::duration<double>(now - s.start).count();
	const double unix_s   = std::chrono::duration<double>(
                              std::chrono::system_clock::now().time_since_epoch())
	                          .count();
	const uint64_t calls  = g_calls.exchange(0, std::memory_order_relaxed);
	const auto     frames = s.frames;

	std::vector<uint32_t> group_order;
	for (uint32_t i = 0; i < s.groups.size(); i++) {
		if (s.groups[i].draws != 0) {
			group_order.push_back(i);
		}
	}
	std::sort(group_order.begin(), group_order.end(),
	          [&](uint32_t a, uint32_t b) { return s.groups[a].draws > s.groups[b].draws; });
	if (group_order.size() > settings.top_groups) {
		group_order.resize(settings.top_groups);
	}
	uint64_t top_draws = 0;
	for (const auto index: group_order) {
		top_draws += s.groups[index].draws;
	}

	std::vector<uint32_t> class_order;
	for (uint32_t i = 0; i < s.classes.size(); i++) {
		if (s.classes[i].draws != 0) {
			class_order.push_back(i);
		}
	}
	const auto active_classes = class_order.size();
	std::sort(class_order.begin(), class_order.end(),
	          [&](uint32_t a, uint32_t b) { return s.classes[a].draws > s.classes[b].draws; });
	if (class_order.size() > settings.top_classes) {
		class_order.resize(settings.top_classes);
	}
	// Also every class a printed group draws into.
	for (const auto index: group_order) {
		const auto class_index = s.groups[index].class_index;
		if (std::find(class_order.begin(), class_order.end(), class_index) == class_order.end()) {
			class_order.push_back(class_index);
		}
	}

	std::string out = fmt::format(
	    "draw-pass-stats: t={:.1f} unix={:.3f} window={:.2f} frames={} fps={:.2f} calls={} "
	    "draws={} calls_f={:.1f} draws_f={:.1f} depth_only_f={:.1f} gpu_indirect_f={:.1f} "
	    "records_f={:.1f} dispatches={} dispatches_f={:.1f} classes={} groups={} top_share={:.1f}\n",
	    t, unix_s, window, frames, window > 0.0 ? static_cast<double>(frames) / window : 0.0, calls,
	    s.draws, PerFrame(calls, frames), PerFrame(s.draws, frames),
	    PerFrame(s.depth_only_draws, frames), PerFrame(s.indirect_draws, frames),
	    PerFrame(s.indirect_records, frames), s.dispatches, PerFrame(s.dispatches, frames),
	    active_classes, s.group_index.size(), Percent(top_draws, s.draws));

	for (const auto index: class_order) {
		auto& record = s.classes[index];
		std::sort(record.sizes.begin(), record.sizes.end(),
		          [](const ViewSize& a, const ViewSize& b) { return a.draws > b.draws; });
		std::string sizes;
		for (const auto& size: record.sizes) {
			if (size.draws == 0) {
				continue;
			}
			sizes += fmt::format("{}{}x{}:{:.1f}", sizes.empty() ? "" : ",", size.width, size.height,
			                     PerFrame(size.draws, frames));
		}
		out += fmt::format(
		    "draw-pass-stats-class: t={:.1f} id=C{} hash={:016x} draws={} draws_f={:.1f} "
		    "share={:.1f} runs_f={:.2f} views_f={:.2f} depth_only={} vp={} | {}\n",
		    t, record.id, record.hash, record.draws, PerFrame(record.draws, frames),
		    Percent(record.draws, s.draws), PerFrame(record.runs, frames),
		    PerFrame(record.views, frames), record.depth_only ? 1 : 0,
		    sizes.empty() ? std::string("-") : sizes, record.text);
	}

	for (const auto index: group_order) {
		const auto& group = s.groups[index];
		out += fmt::format(
		    "draw-pass-stats-group: t={:.1f} id=G{} class=C{} vs={:#x} vs_hash={:016x} ps={:#x} "
		    "ps_hash={} draws={} draws_f={:.1f} share={:.1f} inst_avg={:.2f} idx_avg={:.0f} "
		    "indirect_f={:.2f} records_f={:.2f}{}\n",
		    t, group.id, s.classes[group.class_index].id, group.vs_address, group.vs_hash,
		    group.ps_address,
		    group.ps_active ? fmt::format("{:016x}", group.ps_hash) : std::string("off"), group.draws,
		    PerFrame(group.draws, frames), Percent(group.draws, s.draws),
		    group.draws != 0
		        ? static_cast<double>(group.instances) / static_cast<double>(group.draws)
		        : 0.0,
		    group.direct_draws != 0 ? static_cast<double>(group.direct_indices) /
		                                  static_cast<double>(group.direct_draws)
		                            : 0.0,
		    PerFrame(group.indirect_draws, frames), PerFrame(group.indirect_records, frames),
		    group.mesh ? " mesh" : "");
	}

	std::vector<uint32_t> compute_order;
	for (uint32_t i = 0; i < s.computes.size(); i++) {
		if (s.computes[i].count != 0) {
			compute_order.push_back(i);
		}
	}
	std::sort(compute_order.begin(), compute_order.end(),
	          [&](uint32_t a, uint32_t b) { return s.computes[a].count > s.computes[b].count; });
	if (compute_order.size() > settings.top_cs) {
		compute_order.resize(settings.top_cs);
	}
	for (const auto index: compute_order) {
		const auto&    cs     = s.computes[index];
		const uint64_t direct = cs.count - cs.indirect;
		out += fmt::format("draw-pass-stats-cs: t={:.1f} cs={:#x} hash={:016x} n={} n_f={:.1f} "
		                   "share={:.1f} groups_avg={:.1f} tiny_f={:.1f} indirect_f={:.1f}\n",
		                   t, cs.address, cs.hash, cs.count, PerFrame(cs.count, frames),
		                   Percent(cs.count, s.dispatches),
		                   direct != 0 ? static_cast<double>(cs.groups) / static_cast<double>(direct)
		                               : 0.0,
		                   PerFrame(cs.tiny, frames), PerFrame(cs.indirect, frames));
	}
	std::fputs(out.c_str(), stdout);
	std::fflush(stdout);

	for (auto& record: s.classes) {
		record.draws = record.runs = record.views = 0;
		for (auto& size: record.sizes) {
			size.draws = 0;
		}
	}
	for (auto& group: s.groups) {
		group.draws = group.instances = group.direct_draws = group.direct_indices = 0;
		group.indirect_draws = group.indirect_records = 0;
	}
	for (auto& cs: s.computes) {
		cs.count = cs.groups = cs.tiny = cs.indirect = 0;
	}
	s.frames = s.draws = s.depth_only_draws = s.indirect_draws = s.indirect_records = 0;
	s.dispatches   = 0;
	s.window_start = now;
}

// KYTY_SKIP_DRAW_* (see drawPassStats.h). Measurement only.
struct SkipSettings {
	std::vector<uint64_t> ps;
	std::vector<uint64_t> vs;
	std::vector<uint64_t> classes;
	std::vector<uint64_t> sizes; // width << 32 | height
	uint32_t              keep_views = 0;
	bool                  color      = false;
	int32_t               ab_bit     = -1;
	double                after_s    = 0.0;
	bool                  any        = false;
	// Only PS/VS rules: decidable before the programs are looked up.
	bool early = false;
};

std::vector<uint64_t> EnvList(const char* name, bool sizes) {
	std::vector<uint64_t> list;
	const char*           value = std::getenv(name);
	if (value == nullptr) {
		return list;
	}
	const std::string text(value);
	size_t            begin = 0;
	while (begin < text.size()) {
		const auto end = text.find_first_of(";|+, \t", begin);
		auto token = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
		begin      = end == std::string::npos ? text.size() : end + 1;
		if (!token.empty() && token[0] == '#') {
			token.erase(0, 1);
		}
		if (token.empty()) {
			continue;
		}
		if (sizes) {
			const auto x = token.find_first_of("xX");
			if (x == std::string::npos) {
				continue;
			}
			const uint64_t width  = std::strtoull(token.substr(0, x).c_str(), nullptr, 10);
			const uint64_t height = std::strtoull(token.substr(x + 1).c_str(), nullptr, 10);
			list.push_back((width << 32u) | height);
		} else {
			list.push_back(std::strtoull(token.c_str(), nullptr, 16));
		}
	}
	return list;
}

SkipSettings ReadSkipSettings() {
	SkipSettings settings;
	settings.ps         = EnvList("KYTY_SKIP_DRAW_PS", false);
	settings.vs         = EnvList("KYTY_SKIP_DRAW_VS", false);
	settings.classes    = EnvList("KYTY_SKIP_DRAW_CLASS", false);
	settings.sizes      = EnvList("KYTY_SKIP_DRAW_VP", true);
	settings.keep_views = EnvUint("KYTY_SKIP_DRAW_KEEP_VIEWS", 0);
	settings.color      = EnvFlag("KYTY_SKIP_DRAW_COLOR");
	if (const char* value = std::getenv("KYTY_SKIP_DRAW_AB_BIT");
	    value != nullptr && *value != '\0') {
		const int bit   = std::atoi(value);
		settings.ab_bit = bit >= 0 && bit < 63 ? bit : -1;
	}
	if (const char* value = std::getenv("KYTY_SKIP_DRAW_AFTER"); value != nullptr) {
		settings.after_s = std::max(std::atof(value), 0.0);
	}
	settings.any   = !settings.ps.empty() || !settings.vs.empty() || !settings.classes.empty() ||
	                 !settings.sizes.empty();
	settings.early = settings.any && settings.classes.empty() && settings.sizes.empty() &&
	                 settings.keep_views == 0;
	return settings;
}

const SkipSettings& GetSkipSettings() {
	static const SkipSettings settings = ReadSkipSettings();
	return settings;
}

bool Contains(const std::vector<uint64_t>& list, uint64_t value) {
	return std::find(list.begin(), list.end(), value) != list.end();
}

// What the draws of one VS/PS address pair showed once their programs were looked up.
struct SkipVerdict {
	bool writes = false;
	bool color  = false;
};

struct SkipState {
	std::mutex                                mutex;
	std::unordered_map<uint64_t, SkipVerdict> verdicts;
	std::vector<uint64_t>                     frame_views;
	bool                                      announced = false;
	bool                                      started   = false;
	Clock::time_point                         start;
	Clock::time_point                         window_start;
	uint64_t                                  frames       = 0;
	uint64_t                                  armed_frames = 0;
	uint64_t                                  early        = 0;
	uint64_t                                  late         = 0;
	uint64_t                                  kept_writes  = 0;
	uint64_t                                  kept_color   = 0;
	uint64_t                                  kept_view    = 0;
	uint64_t                                  kept_special = 0;
};

SkipState& GetSkipState() {
	static SkipState* state = new SkipState;
	return *state;
}

// Set at flips once KYTY_SKIP_DRAW_AFTER has passed.
std::atomic<bool> g_skip_time_ok {false};

bool SkipArmed(const SkipSettings& settings) {
	if (!g_skip_time_ok.load(std::memory_order_relaxed)) {
		return false;
	}
	return settings.ab_bit < 0 || Common::AbSwitch::Enabled(static_cast<uint32_t>(settings.ab_bit));
}

// Clears, depth/stencil copies, decompressions and resolves are never skipped.
bool SpecialDraw(const CommandBuffer& buffer) {
	const auto& hw      = buffer.GetRegisters();
	const auto& control = hw.GetRenderControl();
	const auto  mode    = hw.GetColorControl().mode;
	return control.depth_clear_enable || control.stencil_clear_enable ||
	       control.resummarize_enable || control.copy_depth_to_color ||
	       control.copy_stencil_to_color || (mode != 0 && mode != 1);
}

uint64_t GuestVsAddress(const CommandBuffer& buffer) {
	const auto& vs = buffer.GetShaders().GetVs();
	return vs.gs_regs.data_addr != 0 ? vs.gs_regs.data_addr : vs.es_regs.data_addr;
}

bool AddressRulesMatch(const SkipSettings& settings, uint64_t vs_address, uint64_t ps_address) {
	return (settings.ps.empty() || Contains(settings.ps, ps_address)) &&
	       (settings.vs.empty() || Contains(settings.vs, vs_address));
}

std::string ListText(const std::vector<uint64_t>& list, bool sizes) {
	std::string text;
	for (const auto value: list) {
		text += text.empty() ? "" : ";";
		text += sizes ? fmt::format("{}x{}", value >> 32u, value & 0xffffffffu)
		              : fmt::format("{:x}", value);
	}
	return text.empty() ? std::string("-") : text;
}

void AnnounceSkip(SkipState& s, const SkipSettings& settings) {
	if (s.announced) {
		return;
	}
	s.announced = true;
	std::printf("draw-skip: MEASUREMENT ONLY - skipped draws change the rendering; the FPS of this "
	            "run is not an emulator FPS result\n");
	std::printf("draw-skip: ps=%s vs=%s class=%s vp=%s keep_views=%u color=%d ab_bit=%d "
	            "after=%.1f early=%d\n",
	            ListText(settings.ps, false).c_str(), ListText(settings.vs, false).c_str(),
	            ListText(settings.classes, false).c_str(), ListText(settings.sizes, true).c_str(),
	            settings.keep_views, settings.color ? 1 : 0, settings.ab_bit, settings.after_s,
	            settings.early ? 1 : 0);
	std::fflush(stdout);
}

void SkipFlip(double period_s) {
	const auto& settings = GetSkipSettings();
	if (!settings.any) {
		return;
	}
	auto&            s = GetSkipState();
	std::scoped_lock lock(s.mutex);
	AnnounceSkip(s, settings);
	const auto now = Clock::now();
	if (!s.started) {
		s.started      = true;
		s.start        = now;
		s.window_start = now;
	}
	g_skip_time_ok.store(std::chrono::duration<double>(now - s.start).count() >= settings.after_s,
	                     std::memory_order_relaxed);
	s.frame_views.clear();
	s.frames++;
	if (SkipArmed(settings)) {
		s.armed_frames++;
	}
	const double window = std::chrono::duration<double>(now - s.window_start).count();
	if (window < period_s) {
		return;
	}
	const double unix_s =
	    std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
	std::printf("draw-skip: t=%.1f unix=%.3f window=%.2f frames=%llu armed_frames=%llu "
	            "early_f=%.1f late_f=%.1f kept_writes_f=%.1f kept_color_f=%.1f kept_view_f=%.1f "
	            "kept_special_f=%.1f\n",
	            std::chrono::duration<double>(now - s.start).count(), unix_s, window,
	            static_cast<unsigned long long>(s.frames),
	            static_cast<unsigned long long>(s.armed_frames), PerFrame(s.early, s.frames),
	            PerFrame(s.late, s.frames), PerFrame(s.kept_writes, s.frames),
	            PerFrame(s.kept_color, s.frames), PerFrame(s.kept_view, s.frames),
	            PerFrame(s.kept_special, s.frames));
	std::fflush(stdout);
	s.frames = s.armed_frames = s.early = s.late = 0;
	s.kept_writes = s.kept_color = s.kept_view = s.kept_special = 0;
	s.window_start = now;
}

} // namespace

bool ReadActive() noexcept {
	return GetSettings().stats || GetSkipSettings().any;
}

void NoteCall() noexcept {
	g_calls.fetch_add(1, std::memory_order_relaxed);
}

void NoteDraw(const CommandBuffer& buffer, const DrawRenderState& state, const DrawTargets& targets,
              uint32_t index_count, uint32_t instance_count, uint32_t indirect_records, bool mesh) {
	if (!GetSettings().stats) {
		return;
	}
	auto&            s = GetState();
	std::scoped_lock lock(s.mutex);
	StartIfNeeded(s);
	const auto  vp          = GetViewport(buffer);
	const auto  class_hash  = ClassHash(targets);
	const auto  class_index = FindClass(s, class_hash, targets);
	auto&       record      = s.classes[class_index];
	record.draws++;
	CountViewSize(record, vp);
	const auto run_key = RunKey(class_hash, targets, vp);
	if (run_key != s.last_run_key) {
		s.last_run_key = run_key;
		record.runs++;
		if (std::find(record.frame_views.begin(), record.frame_views.end(), run_key) ==
		        record.frame_views.end() &&
		    record.frame_views.size() < FRAME_VIEWS_MAX) {
			record.frame_views.push_back(run_key);
		}
		if (!record.touched) {
			record.touched = true;
			s.touched.push_back(class_index);
		}
	}

	const auto& shaders    = buffer.GetShaders();
	const auto& vs         = shaders.GetVs();
	const auto  vs_address = vs.gs_regs.data_addr != 0 ? vs.gs_regs.data_addr : vs.es_regs.data_addr;
	const auto  ps_address = shaders.GetPs().ps_regs.data_addr;
	const auto* vs_module  = state.vertex_info[0].stage.module;
	const auto* ps_module  = state.ps_active ? state.ps_input_info.stage.module : nullptr;
	const auto  vs_hash    = vs_module != nullptr ? vs_module->info.shader_hash : 0;
	const auto  ps_hash    = ps_module != nullptr ? ps_module->info.shader_hash : 0;
	uint64_t    group_hash = Mix(class_hash, vs_address);
	group_hash             = Mix(group_hash, vs_hash);
	group_hash             = Mix(group_hash, ps_address);
	group_hash             = Mix(group_hash, ps_hash);
	group_hash             = Mix(group_hash, (state.ps_active ? 1u : 0u) | (mesh ? 2u : 0u));
	uint32_t group_index   = s.last_group;
	if (group_index == UINT32_MAX || s.last_group_hash != group_hash) {
		auto [it, inserted] =
		    s.group_index.try_emplace(group_hash, static_cast<uint32_t>(s.groups.size()));
		if (inserted) {
			GroupRecord group;
			group.id          = static_cast<uint32_t>(s.groups.size());
			group.class_index = class_index;
			group.vs_address  = vs_address;
			group.vs_hash     = vs_hash;
			group.ps_address  = ps_address;
			group.ps_hash     = ps_hash;
			group.ps_active   = state.ps_active;
			group.mesh        = mesh;
			s.groups.push_back(group);
		}
		group_index       = it->second;
		s.last_group      = group_index;
		s.last_group_hash = group_hash;
	}
	auto& group = s.groups[group_index];
	group.draws++;
	group.instances += instance_count;
	if (indirect_records != 0) {
		group.indirect_draws++;
		group.indirect_records += indirect_records;
		s.indirect_draws++;
		s.indirect_records += indirect_records;
	} else {
		group.direct_draws++;
		group.direct_indices += index_count;
	}
	s.draws++;
	if (targets.color_count == 0) {
		s.depth_only_draws++;
	}
}

void NoteDispatch(uint64_t cs_address, uint64_t cs_hash, uint32_t x, uint32_t y, uint32_t z,
                  bool indirect) {
	if (!GetSettings().stats) {
		return;
	}
	auto&            s = GetState();
	std::scoped_lock lock(s.mutex);
	StartIfNeeded(s);
	auto [it, inserted] = s.compute_index.try_emplace(Mix(cs_hash, cs_address),
	                                                  static_cast<uint32_t>(s.computes.size()));
	if (inserted) {
		ComputeRecord record;
		record.address = cs_address;
		record.hash    = cs_hash;
		s.computes.push_back(record);
	}
	auto& record = s.computes[it->second];
	record.count++;
	if (indirect) {
		record.indirect++;
	} else {
		const uint64_t groups = uint64_t {x} * y * z;
		record.groups += groups;
		if (groups == 1) {
			record.tiny++;
		}
	}
	s.dispatches++;
}

void NoteFlip() {
	const auto& settings = GetSettings();
	SkipFlip(settings.period_s);
	if (!settings.stats) {
		return;
	}
	auto&            s = GetState();
	std::scoped_lock lock(s.mutex);
	StartIfNeeded(s);
	s.frames++;
	for (const auto index: s.touched) {
		auto& record = s.classes[index];
		record.views += record.frame_views.size();
		record.frame_views.clear();
		record.touched = false;
	}
	s.touched.clear();
	// A new frame starts a new view even if it draws into the same one first.
	s.last_run_key = 0;
	const auto now = Clock::now();
	if (std::chrono::duration<double>(now - s.window_start).count() >= settings.period_s) {
		Report(s, now);
	}
}

bool SkipBeforePrograms(const CommandBuffer& buffer) {
	const auto& settings = GetSkipSettings();
	if (!settings.early || !SkipArmed(settings)) {
		return false;
	}
	const auto vs_address = GuestVsAddress(buffer);
	const auto ps_address = buffer.GetShaders().GetPs().ps_regs.data_addr;
	if (!AddressRulesMatch(settings, vs_address, ps_address) || SpecialDraw(buffer)) {
		return false;
	}
	auto&            s = GetSkipState();
	std::scoped_lock lock(s.mutex);
	const auto       it = s.verdicts.find(Mix(vs_address, ps_address));
	if (it == s.verdicts.end() || it->second.writes || (it->second.color && !settings.color)) {
		return false;
	}
	s.early++;
	return true;
}

bool SkipResolved(const CommandBuffer& buffer, const DrawRenderState& state,
                  const DrawTargets& targets) {
	const auto& settings = GetSkipSettings();
	if (!settings.any) {
		return false;
	}
	const auto vs_address = GuestVsAddress(buffer);
	const auto ps_address = buffer.GetShaders().GetPs().ps_regs.data_addr;
	if (!AddressRulesMatch(settings, vs_address, ps_address)) {
		return false;
	}
	bool writes = false;
	for (const auto& stage: state.VertexStages()) {
		writes = writes || (stage.stage && HasShaderBufferWrites(stage.stage));
	}
	if (state.ps_active && state.ps_input_info.stage) {
		writes = writes || HasShaderBufferWrites(state.ps_input_info.stage);
	}
	const bool       color = targets.color_count != 0;
	auto&            s     = GetSkipState();
	std::scoped_lock lock(s.mutex);
	// Learned in every phase, so that armed phases can skip before the program lookup.
	auto& verdict  = s.verdicts[Mix(vs_address, ps_address)];
	verdict.writes = verdict.writes || writes;
	verdict.color  = verdict.color || color;
	if (!SkipArmed(settings)) {
		return false;
	}
	if (writes) {
		s.kept_writes++;
		return false;
	}
	if (color && !settings.color) {
		s.kept_color++;
		return false;
	}
	if (SpecialDraw(buffer) || targets.depth.depth_clear_enable ||
	    targets.depth.stencil_clear_enable) {
		s.kept_special++;
		return false;
	}
	const auto class_hash = ClassHash(targets);
	if (!settings.classes.empty() && !Contains(settings.classes, class_hash)) {
		return false;
	}
	const auto vp = GetViewport(buffer);
	if (!settings.sizes.empty() &&
	    !Contains(settings.sizes, (uint64_t {vp.width} << 32u) | vp.height)) {
		return false;
	}
	if (settings.keep_views != 0) {
		const auto view  = RunKey(class_hash, targets, vp);
		auto       index = static_cast<size_t>(
            std::find(s.frame_views.begin(), s.frame_views.end(), view) - s.frame_views.begin());
		if (index == s.frame_views.size()) {
			s.frame_views.push_back(view);
		}
		if (index < settings.keep_views) {
			s.kept_view++;
			return false;
		}
	}
	s.late++;
	return true;
}

} // namespace Libs::Graphics::DrawPassStats
