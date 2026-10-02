import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const [entry, file] = process.argv.slice(2);
const { parseGCode, GCodeAnalyzer, GCodeValidator } = await import(pathToFileURL(entry));
const content = readFileSync(file, 'utf8');
const parseErrors = [];
const { commands } = parseGCode(content, {
  includeComments: false,
  onParseError: (error, line) => parseErrors.push({ line, message: error.message }),
});
const axisBC = content.split(/\r?\n/).filter(line => /^\s*G[01]\b/i.test(line) && /\b[BC][-+]?\d/i.test(line)).length;
const config = { nozzleDiameter: 0.4, filamentDiameter: 1.75,
  maxFeedrate: Number.MAX_SAFE_INTEGER,
  bedSize: [Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER],
  maxHeight: Number.MAX_SAFE_INTEGER };
const analysis = new GCodeAnalyzer(config).analyze(commands);
const issues = new GCodeValidator(config, { requireHoming: false,
  checkBedBoundaries: false, checkRapidWithExtrusion: false,
  maxFeedrate: Number.MAX_SAFE_INTEGER,
  minFirstLayerHeight: 0, maxFirstLayerHeight: Number.MAX_SAFE_INTEGER,
  maxHotendTemp: Number.MAX_SAFE_INTEGER,
  maxBedTemp: Number.MAX_SAFE_INTEGER }).validate(commands);
console.log(JSON.stringify({ status: parseErrors.length ? 'PARSE_ERROR' : 'OK',
  parsed: parseErrors.length === 0 && commands.length > 0,
  command_count: commands.length, parse_errors: parseErrors.slice(0, 30),
  stats: analysis.stats, issues: issues.slice(0, 30), issue_count: issues.length,
  unsupported_axes: axisBC ? ['B', 'C'] : [],
  rotary_lines: axisBC,
  limitation: 'Parser ignores B/C; machine-dependent limits disabled because REP5X profile is unverified' }));
