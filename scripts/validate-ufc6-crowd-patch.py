"""Read-only validation of the crowd candidates against the inspected 01.005 SELF.

Does not patch, launch or deploy anything. Rendering and animation still require
live correctness/performance validation. The full original bytes are also kept
in each JSON so the emulator checks them before applying an enabled mod.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


GAME_SHA256 = "afb9e39347b20b9bf627b2a71764ffb33f9e21079951dff0abef24dbb2887a35"
BRANCHES = {
    0x4A27305: ("crowdFillBuffers", 0x4A27652, "0f8447030000"),
    0x4A27CCB: ("crowdIndirectLght", 0x4A27E24, "0f8453010000"),
    0x4A280DC: ("crowdCullDrawVw", 0x4A2828D, "0f84ab010000"),
    0x4A29BDC: ("crowdEsCullBatchPerView", 0x4A29D82, "0f84a0010000"),
}
FILL_ANIM = bytes.fromhex("488bbf200300008bb680620000488b07488b4038ffe0")
FRESH_OUTPUT = {
    0x4A28265: ("crowdCullDrawVw", 0x4A2828D, "488b75c0c7460800000000eb1b", 0x4A28229),
    0x4A29D65: ("crowdEsCullBatchPerView", 0x4A29D82, "488b75c0c7460800000000eb10", 0x4A29D29),
}
FRESH_OFF = bytes.fromhex("4c8b75c04889cf4889cb4c89f6")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def self_reader(blob):
    count = struct.unpack_from("<H", blob, 24)[0]
    elf_pos = 32 + 32 * count
    require(blob[elf_pos:elf_pos + 4] == b"\x7fELF", "Unsupported SELF container")
    headers = [struct.unpack_from("<QQQQ", blob, 32 + 32 * i) for i in range(count)]
    elf = struct.unpack_from("<16sHHIQQQIHHHHHH", blob, elf_pos)
    require(elf[0][:6] == b"\x7fELF\x02\x01" and elf[2] == 62, "Expected x86-64 little-endian ELF")
    programs = [struct.unpack_from("<IIQQQQQQ", blob, elf_pos + elf[5] + elf[9] * i)
                for i in range(elf[10])]

    def read(address, size):
        for index, program in enumerate(programs):
            kind, _, _, vaddr, _, file_size, _, _ = program
            if kind != 1 or not (vaddr <= address and address + size <= vaddr + file_size):
                continue
            for flags, offset, compressed, decoded in headers:
                if flags & 0x800 and ((flags >> 20) & 0xFFF) == index:
                    require(compressed == decoded == file_size, "Compressed/encrypted segment unsupported")
                    begin = offset + address - vaddr
                    result = blob[begin:begin + size]
                    require(len(result) == size, "Truncated SELF segment")
                    return result
        raise ValueError(f"RVA 0x{address:x} is not file backed")
    return read


def validate(game, candidate_path, baseline_path):
    blob = game.read_bytes()
    require(hashlib.sha256(blob).hexdigest() == GAME_SHA256,
            "Game differs from inspected 01.005.000; remap/review before use")
    read = self_reader(blob)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    require(candidate["id"] == "PPSA23566" and candidate["version"] == "01.005.000",
            "Candidate title/version mismatch")
    require(candidate["process"] == "eboot.bin", "Unexpected target process")
    require(len(baseline["mods"]) == 1 and candidate["mods"][0] == baseline["mods"][0],
            "Existing timing mod was not preserved exactly")
    sites = {}
    for mod in candidate["mods"][1:]:
        require(not mod["enabled"], "Tracked crowd candidates must be disabled by default")
        for site in mod["memory"]:
            address = int(site["offset"], 16)
            require(address not in sites, "Duplicate patch site")
            sites[address] = site
    fresh_output = set(sites) == set(FRESH_OUTPUT)
    require(fresh_output or
            (set(BRANCHES) <= set(sites) and set(sites) <= set(BRANCHES) | {0x4A31260}),
            "Unexpected or missing crowd patch sites")
    verified = []
    if fresh_output:
        for address, (job, target, replacement, reset) in FRESH_OUTPUT.items():
            site = sites[address]
            off, on = bytes.fromhex(site["off"]), bytes.fromhex(site["on"])
            require(off == FRESH_OFF == read(address, 13), f"Argument setup mismatch: {job}")
            require(on == bytes.fromhex(replacement), f"Unexpected replacement: {job}")
            require(on[:11] == bytes.fromhex("488b75c0c7460800000000") and on[11] == 0xEB,
                    f"Mirrored output reset mismatch: {job}")
            require(address + 13 + struct.unpack_from("<b", on, 12)[0] == target,
                    f"Replacement branch target mismatch: {job}")
            require(read(reset, 9) == bytes.fromhex("41c744242000000000"),
                    f"Retained per-view count reset mismatch: {job}")
            verified.append({"job": job, "rva": hex(address), "target": hex(target),
                             "retained_view_count_reset": hex(reset),
                             "mirrored_view_argument_count_zeroed": True})
        return {"candidate": str(candidate_path), "game_sha256": GAME_SHA256,
                "timing_preserved": True, "crowd_disabled_by_default": True,
                "variant": "fresh_output", "verified_sites": verified, "live_tested": False}
    for address, (job, target, expected) in BRANCHES.items():
        site = sites[address]
        off, on = bytes.fromhex(site["off"]), bytes.fromhex(site["on"])
        require(off == bytes.fromhex(expected) == read(address, 6), f"Original bytes mismatch: {job}")
        require(read(address - 7, 7) == bytes.fromhex("83bf180e000000"), f"Count guard mismatch: {job}")
        require(on == b"\x90\xe9" + off[2:], f"Unexpected replacement: {job}")
        require(address + 6 + struct.unpack_from("<i", off, 2)[0] == target,
                f"Original branch target mismatch: {job}")
        require(address + 6 + struct.unpack_from("<i", on, 2)[0] == target,
                f"Replacement branch target mismatch: {job}")
        verified.append({"job": job, "rva": hex(address), "target": hex(target)})
    if 0x4A31260 in sites:
        site = sites[0x4A31260]
        require(bytes.fromhex(site["off"]) == FILL_ANIM == read(0x4A31260, 22), "FillAnim thunk mismatch")
        require(bytes.fromhex(site["on"]) == b"\xc3" + FILL_ANIM[1:], "FillAnim replacement mismatch")
        require(read(0x4A31056, 7) == bytes.fromhex("488d3503020000"), "Callback registration mismatch")
        verified.append({"job": "crowdFillAnim", "rva": "0x4a31260", "runtime_validation_required": True})
    return {"candidate": str(candidate_path), "game_sha256": GAME_SHA256,
            "timing_preserved": True, "crowd_disabled_by_default": True,
            "verified_sites": verified, "live_tested": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="Installed timing-only patch JSON")
    args = parser.parse_args()
    print(json.dumps(validate(args.game, args.patch, args.baseline), indent=2))
