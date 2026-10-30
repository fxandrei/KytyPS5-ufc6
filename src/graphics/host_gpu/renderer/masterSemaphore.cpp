#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/abSwitch.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"

#include <chrono>
#include <immintrin.h>
#include <optional>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	if (result != vk::Result::eSuccess && m_stall_handler) {
		m_stall_handler(KnownGpuTick() + 1); // The device is lost: report its last checkpoints.
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::GpuWaits,
	                                         Common::DebugCounters::Counter::GpuWaitNs);
	std::optional<Common::DebugCounters::ScopedTimer> gpu_thread_timer;
	if (GuestGpu::IsGpuThread()) {
		gpu_thread_timer.emplace(Common::DebugCounters::Counter::GpuThreadWaits,
		                         Common::DebugCounters::Counter::GpuThreadWaitNs);
		// The command processor mostly waits here for a few hundred microseconds of GPU work
		// that reads back what a shader just wrote; polling the timeline avoids the wake-up
		// latency of a blocking wait (AbSwitch SPIN_GPU_WAIT, up to 2 ms). Every poll is a driver
		// call and a clock read: polled every eight rounds of pauses, it stays within ~20 us of the
		// GPU while costing an eighth of the calls (AbSwitch SPIN_GPU_WAIT_TUNE).
		if (Common::AbSwitch::Enabled(Common::AbSwitch::SPIN_GPU_WAIT)) {
			const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
			if (Common::AbSwitch::Enabled(Common::AbSwitch::SPIN_GPU_WAIT_TUNE)) {
				uint32_t      polls = 0;
				constexpr int PauseRounds = 128;
				do {
					for (int i = 0; i < PauseRounds; i++) {
						_mm_pause();
					}
					if ((polls++ & 7u) == 0) {
						Refresh();
						if (IsFree(tick)) {
							return;
						}
					}
				} while (std::chrono::steady_clock::now() < deadline);
			} else {
				do {
					for (int i = 0; i < 16; i++) {
						_mm_pause();
					}
					Refresh();
					if (IsFree(tick)) {
						return;
					}
				} while (std::chrono::steady_clock::now() < deadline);
			}
		}
	}
	auto result = m_graphics.device.waitSemaphores(&wait_info,
	                                               m_stall_handler ? STALL_REPORT_NS : UINT64_MAX);
	if (result == vk::Result::eTimeout) {
		m_stall_handler(tick);
		result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	}
	if (result != vk::Result::eSuccess && m_stall_handler) {
		m_stall_handler(tick); // A lost device may still report its last timers.
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	Refresh();
}

} // namespace Libs::Graphics
