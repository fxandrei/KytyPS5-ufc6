ENTRY(ufc6_timing_hook)
PHDRS
{
    hook PT_LOAD FLAGS(5);
    cave PT_LOAD FLAGS(5);
}
SECTIONS
{
    .hook 0x9000353db : { *(.hook) } :hook
    .cave 0x90f000000 : { *(.cave) } :cave
    return_to_game = 0x9000353e9;
    /DISCARD/ : { *(.note.GNU-stack) }
}
