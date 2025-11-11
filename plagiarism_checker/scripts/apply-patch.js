import { existsSync } from 'fs';
import { execSync } from 'child_process';
import { join } from 'path';

const outputDir = join(process.cwd(), '.mastra', 'output');
const difflibPath = join(outputDir, 'node_modules', 'difflib');

if (existsSync(difflibPath)) {
  console.log('Applying patch to difflib in output directory...');
  try {
    execSync('npx patch-package --patch-dir ../../patches', {
      cwd: outputDir,
      stdio: 'inherit'
    });
    console.log('Patch applied successfully!');
  } catch (error) {
    console.error('Failed to apply patch:', error.message);
    process.exit(1);
  }
} else {
  console.log('difflib not found in output directory, skipping patch.');
}

