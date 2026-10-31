//_____________________________________________________________/\_______________________________________________________________
//==============================================================================================================================
//                    AMD FidelityFX SUPER RESOLUTION [FSR 1] ::: SPATIAL SCALING & EXTRAS - v1.20210629
//
// FidelityFX Super Resolution Sample
//
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//------------------------------------------------------------------------------------------------------------------------------
//
// Self-contained GLSL (Vulkan, #version 450) port of the FSR 1.0 EASU float path
// (edge-adaptive spatial upsampling) from ffx_fsr1.h. The macro/constant-packing
// machinery of ffx_a.h is replaced with plain GLSL types and arithmetic; the math
// and tap layout are unchanged. Only the non-packed 32-bit path (FsrEasuF) is
// ported, which is the accurate reference path.
//
// FsrEasu(src, uv, inputSize) returns the filtered color for a fragment whose
// interpolated `uv` spans the full source image (the input image is assumed to
// fill the whole output rectangle, so the resolve position in input pixels is
// `uv * inputSize - 0.5`, which is equivalent to FsrEasuCon with an input
// viewport equal to the input texture size).
//==============================================================================================================================

vec4 FsrEasuR(vec2 p, sampler2D src) { return textureGather(src, p, 0); }
vec4 FsrEasuG(vec2 p, sampler2D src) { return textureGather(src, p, 1); }
vec4 FsrEasuB(vec2 p, sampler2D src) { return textureGather(src, p, 2); }

void FsrEasuTap(inout vec3 aC, inout float aW, vec2 off, vec2 dir, vec2 len, float lob, float clp,
                vec3 c) {
	vec2 v;
	v.x = off.x * dir.x + off.y * dir.y;
	v.y = off.x * (-dir.y) + off.y * dir.x;
	v *= len;
	float d2 = v.x * v.x + v.y * v.y;
	d2 = min(d2, clp);
	float wB = (2.0 / 5.0) * d2 - 1.0;
	float wA = lob * d2 - 1.0;
	wB *= wB;
	wA *= wA;
	wB = (25.0 / 16.0) * wB - (25.0 / 16.0 - 1.0);
	float w = wB * wA;
	aC += c * w;
	aW += w;
}

void FsrEasuSet(inout vec2 dir, inout float len, vec2 pp, bool biS, bool biT, bool biU, bool biV,
                float lA, float lB, float lC, float lD, float lE) {
	float w = 0.0;
	if (biS) w = (1.0 - pp.x) * (1.0 - pp.y);
	if (biT) w = pp.x * (1.0 - pp.y);
	if (biU) w = (1.0 - pp.x) * pp.y;
	if (biV) w = pp.x * pp.y;
	float dc = lD - lC;
	float cb = lC - lB;
	float lenX = max(abs(dc), abs(cb));
	lenX = 1.0 / lenX;
	float dirX = lD - lB;
	dir.x += dirX * w;
	lenX = clamp(abs(dirX) * lenX, 0.0, 1.0);
	lenX *= lenX;
	len += lenX * w;
	float ec = lE - lC;
	float ca = lC - lA;
	float lenY = max(abs(ec), abs(ca));
	lenY = 1.0 / lenY;
	float dirY = lE - lA;
	dir.y += dirY * w;
	lenY = clamp(abs(dirY) * lenY, 0.0, 1.0);
	lenY *= lenY;
	len += lenY * w;
}

vec3 FsrEasu(sampler2D src, vec2 uv, vec2 inputSize) {
	// Resolve position in input pixel space, matching FsrEasuCon for a full-image viewport.
	vec2 pp = uv * inputSize - vec2(0.5);
	vec2 fp = floor(pp);
	pp -= fp;

	// Gather4 centers, in normalized texture coordinates (upper-left oriented).
	vec2 inv = vec2(1.0) / inputSize;
	vec2 p0 = fp * inv + vec2(inv.x, -inv.y);
	vec2 p1 = p0 + vec2(-inv.x, 2.0 * inv.y);
	vec2 p2 = p0 + vec2(inv.x, 2.0 * inv.y);
	vec2 p3 = p0 + vec2(0.0, 4.0 * inv.y);

	vec4 bczzR = FsrEasuR(p0, src);
	vec4 bczzG = FsrEasuG(p0, src);
	vec4 bczzB = FsrEasuB(p0, src);
	vec4 ijfeR = FsrEasuR(p1, src);
	vec4 ijfeG = FsrEasuG(p1, src);
	vec4 ijfeB = FsrEasuB(p1, src);
	vec4 klhgR = FsrEasuR(p2, src);
	vec4 klhgG = FsrEasuG(p2, src);
	vec4 klhgB = FsrEasuB(p2, src);
	vec4 zzonR = FsrEasuR(p3, src);
	vec4 zzonG = FsrEasuG(p3, src);
	vec4 zzonB = FsrEasuB(p3, src);

	// Simplest multi-channel approximate luma (luma times 2).
	vec4 bczzL = bczzB * 0.5 + (bczzR * 0.5 + bczzG);
	vec4 ijfeL = ijfeB * 0.5 + (ijfeR * 0.5 + ijfeG);
	vec4 klhgL = klhgB * 0.5 + (klhgR * 0.5 + klhgG);
	vec4 zzonL = zzonB * 0.5 + (zzonR * 0.5 + zzonG);
	float bL = bczzL.x;
	float cL = bczzL.y;
	float iL = ijfeL.x;
	float jL = ijfeL.y;
	float fL = ijfeL.z;
	float eL = ijfeL.w;
	float kL = klhgL.x;
	float lL = klhgL.y;
	float hL = klhgL.z;
	float gL = klhgL.w;
	float oL = zzonL.z;
	float nL = zzonL.w;

	// Accumulate direction and length from the 2x2 bilinear neighborhood.
	vec2 dir = vec2(0.0);
	float len = 0.0;
	FsrEasuSet(dir, len, pp, true, false, false, false, bL, eL, fL, gL, jL);
	FsrEasuSet(dir, len, pp, false, true, false, false, cL, fL, gL, hL, kL);
	FsrEasuSet(dir, len, pp, false, false, true, false, fL, iL, jL, kL, nL);
	FsrEasuSet(dir, len, pp, false, false, false, true, gL, jL, kL, lL, oL);

	// Normalize with approximation, and clean up close to zero.
	vec2 dir2 = dir * dir;
	float dirR = dir2.x + dir2.y;
	bool zro = dirR < (1.0 / 32768.0);
	dirR = inversesqrt(dirR);
	dirR = zro ? 1.0 : dirR;
	dir.x = zro ? 1.0 : dir.x;
	dir *= vec2(dirR);
	// Transform from {0 to 2} to {0 to 1} range, and shape with square.
	len = len * 0.5;
	len *= len;
	// Stretch kernel {1.0 vert|horz, to sqrt(2.0) on diagonal}.
	float stretch = (dir.x * dir.x + dir.y * dir.y) / max(abs(dir.x), abs(dir.y));
	// Anisotropic length after rotation.
	vec2 len2 = vec2(1.0 + (stretch - 1.0) * len, 1.0 + (-0.5) * len);
	// Window shifts from +/-sqrt(2.0) to slightly beyond 2.0 based on the edge amount.
	float lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
	float clp = 1.0 / lob;

	// Min/max of the 4 nearest taps (f, g, j, k) for deringing.
	vec3 min4 = min(
	    min(min(vec3(ijfeR.z, ijfeG.z, ijfeB.z), vec3(klhgR.w, klhgG.w, klhgB.w)),
	        vec3(ijfeR.y, ijfeG.y, ijfeB.y)),
	    vec3(klhgR.x, klhgG.x, klhgB.x));
	vec3 max4 = max(
	    max(max(vec3(ijfeR.z, ijfeG.z, ijfeB.z), vec3(klhgR.w, klhgG.w, klhgB.w)),
	        vec3(ijfeR.y, ijfeG.y, ijfeB.y)),
	    vec3(klhgR.x, klhgG.x, klhgB.x));

	// 12-tap accumulation.
	vec3 aC = vec3(0.0);
	float aW = 0.0;
	FsrEasuTap(aC, aW, vec2(0.0, -1.0) - pp, dir, len2, lob, clp, vec3(bczzR.x, bczzG.x, bczzB.x));
	FsrEasuTap(aC, aW, vec2(1.0, -1.0) - pp, dir, len2, lob, clp, vec3(bczzR.y, bczzG.y, bczzB.y));
	FsrEasuTap(aC, aW, vec2(-1.0, 1.0) - pp, dir, len2, lob, clp, vec3(ijfeR.x, ijfeG.x, ijfeB.x));
	FsrEasuTap(aC, aW, vec2(0.0, 1.0) - pp, dir, len2, lob, clp, vec3(ijfeR.y, ijfeG.y, ijfeB.y));
	FsrEasuTap(aC, aW, vec2(0.0, 0.0) - pp, dir, len2, lob, clp, vec3(ijfeR.z, ijfeG.z, ijfeB.z));
	FsrEasuTap(aC, aW, vec2(-1.0, 0.0) - pp, dir, len2, lob, clp, vec3(ijfeR.w, ijfeG.w, ijfeB.w));
	FsrEasuTap(aC, aW, vec2(1.0, 1.0) - pp, dir, len2, lob, clp, vec3(klhgR.x, klhgG.x, klhgB.x));
	FsrEasuTap(aC, aW, vec2(2.0, 1.0) - pp, dir, len2, lob, clp, vec3(klhgR.y, klhgG.y, klhgB.y));
	FsrEasuTap(aC, aW, vec2(2.0, 0.0) - pp, dir, len2, lob, clp, vec3(klhgR.z, klhgG.z, klhgB.z));
	FsrEasuTap(aC, aW, vec2(1.0, 0.0) - pp, dir, len2, lob, clp, vec3(klhgR.w, klhgG.w, klhgB.w));
	FsrEasuTap(aC, aW, vec2(1.0, 2.0) - pp, dir, len2, lob, clp, vec3(zzonR.z, zzonG.z, zzonB.z));
	FsrEasuTap(aC, aW, vec2(0.0, 2.0) - pp, dir, len2, lob, clp, vec3(zzonR.w, zzonG.w, zzonB.w));

	// Normalize and dering.
	return min(max4, max(min4, aC / aW));
}
