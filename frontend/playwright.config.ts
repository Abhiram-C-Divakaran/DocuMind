import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { defineConfig } from '@playwright/test';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const E2E_BACKEND_PORT = process.env.E2E_BACKEND_PORT || '18011';
const E2E_FRONTEND_PORT = process.env.E2E_FRONTEND_PORT || '14173';
const BASE_URL = process.env.E2E_BASE_URL || `http://127.0.0.1:${E2E_FRONTEND_PORT}`;

const ROOT_DIR = path.resolve(__dirname, '..');
const WIN_VENV_PYTHON = path.join(ROOT_DIR, '.venv', 'Scripts', 'python.exe');
const UNIX_VENV_PYTHON = path.join(ROOT_DIR, '.venv', 'bin', 'python');
const PYTHON_BIN =
  process.env.E2E_PYTHON ||
  (fs.existsSync(WIN_VENV_PYTHON)
    ? WIN_VENV_PYTHON
    : fs.existsSync(UNIX_VENV_PYTHON)
      ? UNIX_VENV_PYTHON
      : 'python3');

// Strip proxy environment variables so Playwright/Vite/FastAPI localhost checks do not route through a sandbox proxy.
const CLEAN_ENV: Record<string, string> = {};
for (const [k, v] of Object.entries(process.env)) {
  if (
    v !== undefined &&
    !['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY'].includes(k)
  ) {
    CLEAN_ENV[k] = v;
  }
}
CLEAN_ENV.NO_PROXY = '127.0.0.1,localhost';
CLEAN_ENV.no_proxy = '127.0.0.1,localhost';

export default defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  expect: {
    timeout: 15_000,
  },
  fullyParallel: false,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: BASE_URL,
    channel: process.env.PLAYWRIGHT_CHANNEL || (process.platform === 'win32' ? 'chrome' : undefined),
    headless: true,
    viewport: { width: 1440, height: 900 },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
    launchOptions: {
      args: ['--no-proxy-server'],
    },
  },
  webServer: [
    {
      command: `"${PYTHON_BIN}" scripts/e2e_test_server.py`,
      cwd: ROOT_DIR,
      url: `http://127.0.0.1:${E2E_BACKEND_PORT}/health`,
      reuseExistingServer: false,
      timeout: 30_000,
      env: {
        ...CLEAN_ENV,
        E2E_BACKEND_PORT,
      },
    },
    {
      command: `node ./node_modules/vite/bin/vite.js --port ${E2E_FRONTEND_PORT} --host 127.0.0.1`,
      url: BASE_URL,
      reuseExistingServer: false,
      timeout: 30_000,
      env: {
        ...CLEAN_ENV,
        VITE_PROXY_TARGET: `http://127.0.0.1:${E2E_BACKEND_PORT}`,
      },
    },
  ],
});
