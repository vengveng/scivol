// Run once to transfer the compiler-discovered standalone math closure.
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const manifestPath = path.join(root, 'contracts/source_transfers.json');
const hash = data => createHash('sha256').update(data).digest('hex');

if (process.argv.includes('--refresh')) {
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  for (const transfer of manifest.transfers)
    transfer.current_sha256 = hash(readFileSync(path.join(root, transfer.destination)));
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
  process.exit(0);
}

const includeRoots = ['scivol/_native', 'scivol/_csrc'];
const seeds = ['garch_runtime.cpp', 'garch_generated_api.cpp', 'gh_math_translation.cpp']
  .map(name => `scivol/_native/${name}`);
const compiler = process.env.CXX || 'clang++';
const dependencies = execFileSync(compiler,
  ['-std=c++20', '-MM', ...includeRoots.map(dir => `-I${dir}`), ...seeds],
  { cwd: root, encoding: 'utf8' });
const sources = new Set();
// Parse Make dependency records emitted by the preprocessor, not a guessed include graph.
for (const record of dependencies.replace(/\\\r?\n/g, '').trim().split('\n')) {
  const colon = record.indexOf(':');
  if (colon < 0) throw new Error(`Invalid dependency record: ${record}`);
  for (const dependency of record.slice(colon + 1).trim().split(/\s+/))
    sources.add(path.relative(root, path.resolve(root, dependency)));
}

const mathNames = new Set([
  'fixed_differential.hpp', 'student_normalization.hpp', 'student_tail.hpp',
  'runtime_workspace.hpp', 'gh_first_order.hpp', 'gh_equations.hpp',
  'vg_equations.hpp', 'gh_math_translation.cpp',
]);
const destinations = new Map();
for (const source of [...sources].sort()) {
  if (!includeRoots.some(dir => source.startsWith(`${dir}/`)))
    throw new Error(`Unexpected dependency outside the math closure: ${source}`);
  const name = path.basename(source);
  const directory = source.startsWith('scivol/_csrc/') || mathNames.has(name)
    ? 'math' : 'models';
  const destination = `core/src/${directory}/${name}`;
  if ([...destinations.values()].includes(destination))
    throw new Error(`Destination collision: ${destination}`);
  destinations.set(source, destination);
  if (existsSync(path.join(root, destination)))
    throw new Error(`Refusing to overwrite an existing transfer: ${destination}`);
}

const revision = execFileSync('git', ['rev-parse', 'HEAD'],
  { cwd: root, encoding: 'utf8' }).trim();
const manifest = {
  schema_version: 1,
  status: 'prototype/incomplete',
  discovery: { compiler, method: 'C++20 preprocessor Make dependency records', seeds },
  source_revision: revision,
  source_state: 'Working-tree contents, including pre-existing repaired math; hashes are authoritative.',
  transfers: [],
  qualification_debt: [
    'Independent new-core AD and high-precision qualification is pending.',
    'Production binary dispatch, constant metadata, stack frames and paired request timing are pending.',
    'This slice covers only standalone GARCH and GJR-GARCH, not joint means or fitting.',
  ],
};

for (const [source, destination] of destinations) {
  const original = readFileSync(path.join(root, source), 'utf8');
  const includes = [];
  const transferred = original.replace(
    /^(\s*#\s*include\s*)"([^"\n]+)"/gm,
    (directive, prefix, include) => {
      const candidates = [path.join(path.dirname(source), include),
        ...includeRoots.map(dir => path.join(dir, include))];
      const resolved = candidates.map(candidate => path.normalize(candidate))
        .find(candidate => destinations.has(candidate));
      if (!resolved) throw new Error(`Unmapped include in ${source}: ${include}`);
      const target = destinations.get(resolved);
      const relocated = path.relative(path.dirname(destination), target);
      includes.push({ source: include, destination: relocated });
      return `${prefix}"${relocated}"`;
    });
  mkdirSync(path.dirname(path.join(root, destination)), { recursive: true });
  writeFileSync(path.join(root, destination), transferred);
  manifest.transfers.push({ source, destination, source_sha256: hash(original),
    relocated_sha256: hash(transferred), current_sha256: hash(transferred),
    adaptation: 'Literal include relocation only at transfer; subsequent adaptations are tracked in git.',
    includes });
}
mkdirSync(path.dirname(manifestPath), { recursive: true });
writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
console.log(`Transferred ${destinations.size} files; no solver, Python or third-party dependency found.`);
