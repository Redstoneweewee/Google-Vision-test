/**
 * ANSI terminal color constants and helper functions.
 * Centralized here to avoid duplication across modules.
 */

export const RESET  = '\x1b[0m';
export const BOLD   = '\x1b[1m';
export const DIM    = '\x1b[2m';

export const GREEN  = '\x1b[32m';
export const YELLOW = '\x1b[33m';
export const RED    = '\x1b[31m';
export const CYAN   = '\x1b[36m';

export const BG_GREEN  = '\x1b[42m';
export const BG_YELLOW = '\x1b[43m';
export const BG_RED    = '\x1b[41m';

export function colorPass(text: string): string   { return `${GREEN}${text}${RESET}`; }
export function colorWarn(text: string): string   { return `${YELLOW}${text}${RESET}`; }
export function colorError(text: string): string  { return `${RED}${text}${RESET}`; }
export function colorBold(text: string): string   { return `${BOLD}${text}${RESET}`; }
export function colorDim(text: string): string    { return `${DIM}${text}${RESET}`; }
export function colorCyan(text: string): string   { return `${CYAN}${text}${RESET}`; }
