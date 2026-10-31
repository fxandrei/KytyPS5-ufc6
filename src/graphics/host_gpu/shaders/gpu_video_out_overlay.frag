#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 0) out vec4 color;

vec3 SdrToHdr10(vec3 rgb, bool source_linear) {
	rgb = clamp(rgb, 0.0, 1.0);
	vec3 linear = source_linear ? rgb :
	    mix(rgb / 12.92, pow((rgb + 0.055) / 1.055, vec3(2.4)),
	        greaterThan(rgb, vec3(0.04045)));
	// Convert primaries and scale 250-nit reference white relative to 10,000 nits.
	const mat3 to_rec2020 = (250.0 / 10000.0) *
	    mat3(0.627404, 0.069097, 0.016391,
	         0.329283, 0.919540, 0.088013,
	         0.043313, 0.011362, 0.895595);
	vec3 powered = pow(to_rec2020 * linear, vec3(2610.0 / 16384.0));
	return pow((3424.0 / 4096.0 + (2413.0 / 128.0) * powered) /
	               (1.0 + (2392.0 / 128.0) * powered), vec3(2523.0 / 32.0));
}

#ifdef IMGUI_HDR
layout(set = 0, binding = 0) uniform texture2D image;
layout(set = 1, binding = 0) uniform sampler image_sampler;
layout(location = 0) in struct { vec4 Color; vec2 UV; } In;

// Bicubic Catmull-Rom upscale: 16 taps, clamped to the source. Much cheaper than the EASU port
// (which cost ~80 ms/frame at 4K in the first test) and sharper than bilinear.
vec3 CatmullRomSample(sampler2D src, vec2 uv, vec2 size) {
	vec2 pos  = uv * size - 0.5;
	vec2 base = floor(pos);
	vec2 f    = pos - base;
	vec4 wx = vec4(-0.5 * f.x + f.x * f.x - 0.5 * f.x * f.x * f.x,
	               1.0 - 2.5 * f.x * f.x + 1.5 * f.x * f.x * f.x,
	               0.5 * f.x + 2.0 * f.x * f.x - 1.5 * f.x * f.x * f.x,
	               -0.5 * f.x * f.x + 0.5 * f.x * f.x * f.x);
	vec4 wy = vec4(-0.5 * f.y + f.y * f.y - 0.5 * f.y * f.y * f.y,
	               1.0 - 2.5 * f.y * f.y + 1.5 * f.y * f.y * f.y,
	               0.5 * f.y + 2.0 * f.y * f.y - 1.5 * f.y * f.y * f.y,
	               -0.5 * f.y * f.y + 0.5 * f.y * f.y * f.y);
	ivec2 limit = ivec2(size) - 1;
	vec3  sum   = vec3(0.0);
	for (int y = 0; y < 4; ++y) {
		for (int x = 0; x < 4; ++x) {
			ivec2 p = clamp(ivec2(base) + ivec2(x - 1, y - 1), ivec2(0), limit);
			sum += texelFetch(src, p, 0).rgb * wx[x] * wy[y];
		}
	}
	return sum;
}

void main() {
	vec4 sampled = In.Color * texture(sampler2D(image, image_sampler), In.UV);
	color = vec4(SdrToHdr10(sampled.rgb, false), sampled.a);
}
#else
#include "gpu_video_out_hdr10_to_sdr.inc"

layout(location = 0) in vec2 uv;
layout(binding = 0) uniform sampler2D source;
layout(push_constant) uniform AlphaControl {
	uint flags; // bit 0: premultiplied, 1: HDR target, 2: PQ source, 3: linear source, 4: opaque, 5: upscale
};

// Bicubic Catmull-Rom upscale: 16 taps, clamped to the source. Much cheaper than the EASU port
// (which cost ~80 ms/frame at 4K in the first test) and sharper than bilinear.
vec3 CatmullRomSample(sampler2D src, vec2 uv, vec2 size) {
	vec2 pos  = uv * size - 0.5;
	vec2 base = floor(pos);
	vec2 f    = pos - base;
	vec4 wx = vec4(-0.5 * f.x + f.x * f.x - 0.5 * f.x * f.x * f.x,
	               1.0 - 2.5 * f.x * f.x + 1.5 * f.x * f.x * f.x,
	               0.5 * f.x + 2.0 * f.x * f.x - 1.5 * f.x * f.x * f.x,
	               -0.5 * f.x * f.x + 0.5 * f.x * f.x * f.x);
	vec4 wy = vec4(-0.5 * f.y + f.y * f.y - 0.5 * f.y * f.y * f.y,
	               1.0 - 2.5 * f.y * f.y + 1.5 * f.y * f.y * f.y,
	               0.5 * f.y + 2.0 * f.y * f.y - 1.5 * f.y * f.y * f.y,
	               -0.5 * f.y * f.y + 0.5 * f.y * f.y * f.y);
	ivec2 limit = ivec2(size) - 1;
	vec3  sum   = vec3(0.0);
	for (int y = 0; y < 4; ++y) {
		for (int x = 0; x < 4; ++x) {
			ivec2 p = clamp(ivec2(base) + ivec2(x - 1, y - 1), ivec2(0), limit);
			sum += texelFetch(src, p, 0).rgb * wx[x] * wy[y];
		}
	}
	return sum;
}

void main() {
	const uint HDR_TARGET = 2u;
	const uint PQ_SOURCE = 4u;
	const uint UPSCALE = 32u;
	vec4 sampled;
	if ((flags & UPSCALE) != 0u) {
		// Opaque edge-adaptive upscale: the filtered source is the output, alpha 1.
		sampled = vec4(CatmullRomSample(source, uv, vec2(textureSize(source, 0))), 1.0);
	} else {
		sampled = texture(source, uv);
		if ((flags & 16u) != 0u) {
			sampled.a = 1.0;
		}
		bool premultiplied = (flags & 1u) != 0u;
		uint conversion = flags & (HDR_TARGET | PQ_SOURCE);
		bool convert = conversion == HDR_TARGET || conversion == PQ_SOURCE;
		if (convert && premultiplied) {
			sampled.rgb = sampled.a > 0.0 ? sampled.rgb / sampled.a : vec3(0.0);
		}
		if (conversion == HDR_TARGET) {
			sampled.rgb = SdrToHdr10(sampled.rgb, (flags & 8u) != 0u);
		} else if (conversion == PQ_SOURCE) {
			sampled.rgb = Hdr10ToSdr(sampled.rgb, (flags & 8u) != 0u);
		}
		if (!premultiplied || convert) {
			sampled.rgb *= sampled.a;
		}
	}
	color = sampled;
}
#endif
