// Thin re-export shim — implementation moved to packages/web-shared (CR-016).
// Keeps old import paths working. New code should import from 'web-shared'.
export { GexLandscape as default, PANEL_WIDTH, CONFLUENCE_COLORS, QUALITY_SHORT, NEG_COLOR } from 'web-shared'
