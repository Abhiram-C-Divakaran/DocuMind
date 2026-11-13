import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { expect, test } from '@playwright/test';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ADMIN_EMAIL = process.env.E2E_ADMIN_EMAIL || 'admin@documind.e2e';
const ADMIN_PASSWORD = process.env.E2E_ADMIN_PASSWORD || 'AdminStrong123!';
const REVIEWER_EMAIL = 'reviewer@documind.e2e';
const REVIEWER_PASSWORD = 'ReviewerStrong123!';
const VIEWER_EMAIL = 'viewer@documind.e2e';
const VIEWER_PASSWORD = 'ViewerStrong123!';

const ROOT_DIR = path.resolve(__dirname, '../..');
const SAMPLE_RECEIPT = path.join(ROOT_DIR, 'docs', 'demo_sample_receipt.png');
const SCREENSHOT_DIR = path.join(ROOT_DIR, 'docs', 'screenshots');

function ensureScreenshotDir() {
  if (!fs.existsSync(SCREENSHOT_DIR)) {
    fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
  }
}

async function signIn(page: import('@playwright/test').Page, email: string, password: string) {
  await page.fill('#login-email', email);
  await page.fill('#login-password', password);
  await page.getByRole('button', { name: 'Sign In' }).click();
}

test.describe('DocuMind Full End-to-End Application Journey', () => {
  test('admin provisions reviewer & viewer -> reviewer uploads, polls async job, corrects, drafts, approves, checks history -> admin exports feedback -> viewer blocked from mutations', async ({
    page,
  }) => {
    page.on('pageerror', (err) => {
      console.error('BROWSER_PAGEERROR:', err.stack || err.message);
    });
    ensureScreenshotDir();

    if (!fs.existsSync(SAMPLE_RECEIPT)) {
      throw new Error(`Missing synthetic sample receipt fixture at ${SAMPLE_RECEIPT}`);
    }

    // 1. Administrator logs in
    await page.goto('/');
    await expect(page.getByText('Sign in to Workspace')).toBeVisible();
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '01_login.png'), fullPage: true });

    await signIn(page, ADMIN_EMAIL, ADMIN_PASSWORD);
    await expect(page.getByText('Platform Overview')).toBeVisible();

    // Verify no refresh/access token leaked to localStorage or sessionStorage
    const leakedTokens = await page.evaluate(() => ({
      localKeys: Object.keys(window.localStorage),
      sessionKeys: Object.keys(window.sessionStorage),
      cookies: document.cookie,
    }));
    expect(leakedTokens.localKeys).toEqual([]);
    expect(leakedTokens.sessionKeys).toEqual([]);
    expect(leakedTokens.cookies).not.toContain('documind_refresh_token'); // HttpOnly hides it from JS
    expect(leakedTokens.cookies).toContain('documind_csrf_token');

    // 2. Administrator creates a Reviewer and a Viewer
    await page.getByRole('button', { name: 'Admin & Jobs Ops' }).click();
    await expect(page.getByText('Provision User Account')).toBeVisible();

    await page.getByPlaceholder('Email address').fill(REVIEWER_EMAIL);
    await page.getByPlaceholder('Full name (optional)').fill('Alice Reviewer');
    await page.getByPlaceholder('Password (>=8 chars)').fill(REVIEWER_PASSWORD);
    await page.locator('select').selectOption('REVIEWER');
    await page.getByRole('button', { name: 'Create Account' }).click();
    await expect(page.getByText(REVIEWER_EMAIL, { exact: true }).first()).toBeVisible();

    await page.getByPlaceholder('Email address').fill(VIEWER_EMAIL);
    await page.getByPlaceholder('Full name (optional)').fill('Bob Viewer');
    await page.getByPlaceholder('Password (>=8 chars)').fill(VIEWER_PASSWORD);
    await page.locator('select').selectOption('VIEWER');
    await page.getByRole('button', { name: 'Create Account' }).click();
    await expect(page.getByText(VIEWER_EMAIL, { exact: true }).first()).toBeVisible();

    await page.getByRole('button', { name: 'Sign Out' }).click();
    await expect(page.getByText('Sign in to Workspace')).toBeVisible();

    // 3. Reviewer logs in & verify HttpOnly cookie session persists across reload
    await signIn(page, REVIEWER_EMAIL, REVIEWER_PASSWORD);
    await expect(page.getByText('Platform Overview')).toBeVisible();
    await page.reload();
    await expect(page.getByText(REVIEWER_EMAIL, { exact: true }).first()).toBeVisible();

    // 4 & 5. Reviewer uploads synthetic receipt & API returns queued 202 document
    await page.getByRole('button', { name: 'Upload Documents' }).click();
    await expect(page.getByText('Ingest Scanned Document')).toBeVisible();

    const uploadPromise = page.waitForResponse(
      (res) => res.url().endsWith('/api/v1/documents') && res.request().method() === 'POST'
    );
    await page.locator('input[type="file"]').setInputFiles(SAMPLE_RECEIPT);
    await page.getByLabel('Flag for manual human review regardless of confidence threshold').check();
    await page.getByRole('button', { name: 'Submit for Ingestion' }).click();
    const uploadRes = await uploadPromise;
    expect(uploadRes.status()).toBe(202);
    const uploadJson = await uploadRes.json();
    const uploadedDocId: string = uploadJson.document.id;

    // 6 & 7. Async polling updates Document Ledger status to COMPLETED
    await expect(page.getByText('Document Ledger').first()).toBeVisible();
    const docRow = page.locator('tr', { hasText: uploadedDocId });
    await expect(docRow).toBeVisible();
    await expect(docRow.getByText('COMPLETED')).toBeVisible({ timeout: 25_000 });
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '02_document_ledger.png'), fullPage: true });

    // 8. User opens extracted result modal
    await docRow.click();
    await expect(page.getByText(/SUNRISE.*BAKERY/i).first()).toBeVisible();
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '03_extraction_modal.png'), fullPage: true });
    await page.getByRole('button', { name: 'Open Review Workspace' }).click();

    // 9. User enters Review Workspace
    await expect(page.getByText('Structured Receipt Corrections')).toBeVisible();
    await expect(page.getByText('Authenticated Audit Provenance:')).toBeVisible();

    // 10 & 11. User corrects Vendor Name & Reviewer Notes, then saves Draft
    const vendorInput = page.getByPlaceholder('e.g. Acme Coffee Roasters');
    await vendorInput.fill('SUNRISE ARTISAN BAKERY LLC');
    const notesInput = page.locator('textarea');
    await notesInput.fill('Corrected merchant legal suffix from receipt header.');
    await page.getByRole('button', { name: 'Save Draft' }).click();
    await expect(page.getByText(/Draft saved/i).first()).toBeVisible();

    // 12 & 13. User approves the review & checks Audit History
    await page.getByRole('button', { name: 'Approve & Verify' }).click();
    await expect(page.getByText(/Review approved!/i).first()).toBeVisible();
    await expect(page.getByText('Correction Diffs & Audit History')).toBeVisible();
    await expect(page.getByText('Corrected merchant legal suffix from receipt header.').first()).toBeVisible();
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '04_review_workspace.png'), fullPage: true });

    // Sign out reviewer
    await page.getByRole('button', { name: 'Sign Out' }).click();

    // 14. Admin logs in, inspects Job Telemetry & Health, and triggers Feedback Export
    await signIn(page, ADMIN_EMAIL, ADMIN_PASSWORD);
    await expect(page.getByText('Platform Overview')).toBeVisible();
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '05_dashboard_overview.png'), fullPage: true });

    await page.getByRole('button', { name: 'Admin & Jobs Ops' }).click();
    await expect(page.getByText('Queue Operations & Active Learning Export')).toBeVisible();
    await expect(page.getByText('Total Jobs')).toBeVisible();
    await expect(page.getByText('Avg Duration')).toBeVisible();

    const exportPromise = page.waitForResponse(
      (res) => res.url().includes('/api/v1/reviews/feedback-export') && res.request().method() === 'POST'
    );
    await page.getByRole('button', { name: 'Export Approved Feedback' }).click();
    const exportRes = await exportPromise;
    expect(exportRes.status()).toBe(200);
    const exportJson = await exportRes.json();
    expect(exportJson.exported_count).toBeGreaterThanOrEqual(1);
    await expect(page.getByText(/Exported \d+ approved feedback records/i)).toBeVisible();
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '06_admin_jobs_ops.png'), fullPage: true });

    await page.getByRole('button', { name: 'Sign Out' }).click();

    // 15. Viewer logs in and permissions prevent unauthorized uploads/edits
    await signIn(page, VIEWER_EMAIL, VIEWER_PASSWORD);
    await expect(page.getByText('Platform Overview')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Upload Documents' })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Admin & Jobs Ops' })).toHaveCount(0);

    // Verify backend API also strictly rejects Viewer mutations with 403
    const viewerUploadStatus = await page.evaluate(async () => {
      const form = new FormData();
      form.append('file', new Blob(['%PDF-1.4\n'], { type: 'application/pdf' }), 'blocked.pdf');
      const csrfCookie = document.cookie
        .split('; ')
        .find((c) => c.startsWith('documind_csrf_token='))
        ?.split('=')[1];
      const refRes = await fetch('/api/v1/auth/refresh', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          'X-Auth-Transport': 'cookie',
          'X-CSRF-Token': decodeURIComponent(csrfCookie || ''),
        },
        body: '{}',
      });
      const tok = await refRes.json();
      const up = await fetch('/api/v1/documents', {
        method: 'POST',
        headers: { Authorization: `Bearer ${tok.access_token}` },
        body: form,
      });
      return up.status;
    });
    expect(viewerUploadStatus).toBe(403);
  });
});

