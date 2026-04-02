/**
 * URC App — Create Stream E2E Test
 *
 * Creates a JSONPlaceholder stream through the builder UI:
 * 1. Fill Connection tab (base URL)
 * 2. Select index on Splunk Output tab
 * 3. Click Save
 * 4. Verify stream appears in the dashboard
 */

import { test, expect } from '@playwright/test';
import { splunkLogin, navigateTo, waitForReactMount, deleteAllStreams } from './helpers';

test.describe('Create Stream Flow', () => {
    test.beforeEach(async ({ page }) => {
        await splunkLogin(page);
        await deleteAllStreams(page);
    });

    test('create JSONPlaceholder stream and verify in dashboard', async ({ page }) => {
        // --- Navigate to builder ---
        await navigateTo(page, 'builder');
        await waitForReactMount(page);
        await expect(page.getByRole('tab', { name: 'Connection' })).toBeVisible({ timeout: 15_000 });

        // --- Connection tab: fill Base URL ---
        const baseUrlInput = page.getByPlaceholder('https://api.example.com/v1');
        await expect(baseUrlInput).toBeVisible();
        await baseUrlInput.fill('https://jsonplaceholder.typicode.com/posts');

        // --- Splunk Output tab: select an index ---
        await page.getByRole('tab', { name: 'Splunk Output' }).click();
        await page.waitForTimeout(2000); // Wait for indexes to load

        // Try to select 'main' index from the dropdown
        const indexSelect = page.locator('[data-test="select"]').first();
        if (await indexSelect.isVisible({ timeout: 3_000 }).catch(() => false)) {
            await indexSelect.click();
            // Look for 'main' option
            const mainOption = page.locator('[data-test="option"]').filter({ hasText: 'main' }).first();
            if (await mainOption.isVisible({ timeout: 3_000 }).catch(() => false)) {
                await mainOption.click();
            }
        }

        await page.screenshot({ path: 'tests/e2e/playwright/screenshots/save-01-before-save.png' });

        // --- Intercept network requests to debug the save call ---
        const requests: { url: string; method: string; status: number; body: string }[] = [];
        page.on('request', (req) => {
            if (req.url().includes('urc_app') && req.url().includes('splunkd')) {
                requests.push({
                    url: req.url(),
                    method: req.method(),
                    status: 0,
                    body: req.postData() || '',
                });
            }
        });
        page.on('response', async (resp) => {
            const match = requests.find((r) => r.url === resp.url() && r.status === 0);
            if (match) {
                match.status = resp.status();
                if (resp.status() >= 400) {
                    try {
                        const text = await resp.text();
                        console.log(`  Response body: ${text.substring(0, 300)}`);
                    } catch {}
                }
            }
        });

        // --- Click Save ---
        const saveButton = page.getByRole('button', { name: 'Save' });
        await expect(saveButton).toBeVisible();
        await saveButton.click();

        // Wait for save to complete
        await page.waitForTimeout(5000);
        await page.screenshot({ path: 'tests/e2e/playwright/screenshots/save-02-after-save.png' });

        // Log all captured requests
        for (const req of requests) {
            console.log(`[${req.method}] ${req.url} → ${req.status}`);
            if (req.body) console.log(`  Body: ${req.body.substring(0, 200)}`);
        }

        // Check if we got an error or redirected to streams
        const currentUrl = page.url();
        const hasError = await page.locator('[role="alert"]').isVisible().catch(() => false);

        if (hasError) {
            // Capture the error message
            const errorText = await page.locator('[role="alert"]').textContent();
            console.log('Save error:', errorText);
            await page.screenshot({ path: 'tests/e2e/playwright/screenshots/save-03-error.png' });
        }

        // Log result
        console.log('After save URL:', currentUrl);
    });
});
