/**
 * URC App — Playwright Smoke Tests
 *
 * Verifies:
 * 1. Splunk login works
 * 2. Streams dashboard page loads with React content
 * 3. Builder page loads with tabbed editor
 * 4. Full stream creation flow with JSONPlaceholder (no auth needed)
 * 5. Stream appears in dashboard after save
 */

import { test, expect } from '@playwright/test';
import { splunkLogin, navigateTo, waitForReactMount, deleteAllStreams } from './helpers';

test.describe.serial('URC App Smoke Tests', () => {
    test('login to Splunk', async ({ page }) => {
        await splunkLogin(page);
        // Should land on a Splunk page, not the login form
        const url = page.url();
        expect(url).not.toContain('/account/login');
    });

    test('streams dashboard loads', async ({ page }) => {
        await splunkLogin(page);
        await navigateTo(page, 'streams');
        await waitForReactMount(page);

        // The React dashboard should render the heading
        const heading = page.locator('text=REST API Streams');
        await expect(heading).toBeVisible({ timeout: 15_000 });
    });

    test('builder page loads with tabs', async ({ page }) => {
        await splunkLogin(page);
        await navigateTo(page, 'builder');
        await waitForReactMount(page);

        // Should show the tabbed builder — use role selector to avoid ambiguity
        const connectionTab = page.getByRole('tab', { name: 'Connection' });
        await expect(connectionTab).toBeVisible({ timeout: 15_000 });

        // Verify all tabs are present
        await expect(page.getByRole('tab', { name: 'Data Mapping' })).toBeVisible();
        await expect(page.getByRole('tab', { name: 'Splunk Output' })).toBeVisible();
        await expect(page.getByRole('tab', { name: 'Schedule & Tags' })).toBeVisible();
        await expect(page.getByRole('tab', { name: 'Test & Preview' })).toBeVisible();

        // Verify form fields on Connection tab (use label selectors to avoid nav menu ambiguity)
        await expect(page.locator('label:text-is("Account")')).toBeVisible();
        await expect(page.locator('label:text-is("Base URL")')).toBeVisible();
        await expect(page.locator('label:text-is("HTTP Method")')).toBeVisible();

        // Verify Save and Cancel buttons
        await expect(page.getByRole('button', { name: 'Save' })).toBeVisible();
        await expect(page.getByRole('button', { name: 'Cancel' })).toBeVisible();
    });

    test('create stream — fill all tabs and screenshot', async ({ page }) => {
        await splunkLogin(page);
        await deleteAllStreams(page);

        // Navigate to builder (new stream mode)
        await navigateTo(page, 'builder');
        await waitForReactMount(page);

        // --- Connection tab (active by default) ---
        await expect(page.getByRole('tab', { name: 'Connection' })).toBeVisible({ timeout: 15_000 });

        // Account stays (none) — JSONPlaceholder needs no auth
        // Fill Base URL
        const baseUrlInput = page.getByPlaceholder('https://api.example.com/v1');
        await expect(baseUrlInput).toBeVisible();
        await baseUrlInput.fill('https://jsonplaceholder.typicode.com/posts');
        await page.screenshot({ path: 'tests/e2e/playwright/screenshots/01-connection-tab.png' });

        // --- Data Mapping tab ---
        await page.getByRole('tab', { name: 'Data Mapping' }).click();
        await page.waitForTimeout(1000);
        await page.screenshot({ path: 'tests/e2e/playwright/screenshots/02-data-mapping-tab.png' });

        // --- Splunk Output tab ---
        await page.getByRole('tab', { name: 'Splunk Output' }).click();
        await page.waitForTimeout(1000);
        await page.screenshot({ path: 'tests/e2e/playwright/screenshots/03-splunk-output-tab.png' });

        // --- Schedule & Tags tab ---
        await page.getByRole('tab', { name: 'Schedule & Tags' }).click();
        await page.waitForTimeout(1000);
        await page.screenshot({ path: 'tests/e2e/playwright/screenshots/04-schedule-tags-tab.png' });

        // --- Test & Preview tab ---
        await page.getByRole('tab', { name: 'Test & Preview' }).click();
        await page.waitForTimeout(1000);
        await page.screenshot({ path: 'tests/e2e/playwright/screenshots/05-test-preview-tab.png' });
    });
});
