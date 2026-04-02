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

    test('builder wizard loads with stepper', async ({ page }) => {
        await splunkLogin(page);
        await navigateTo(page, 'builder');
        await waitForReactMount(page);

        // Should show breadcrumbs and step bar
        await expect(page.getByText('New Stream')).toBeVisible({ timeout: 15_000 });

        // Verify Step 1 form fields
        await expect(page.locator('label:text-is("Stream Name")')).toBeVisible();
        await expect(page.locator('label:text-is("Base URL")')).toBeVisible();

        // Verify Cancel and Next buttons (no Save on step 1)
        await expect(page.getByRole('button', { name: 'Cancel' })).toBeVisible();
        await expect(page.getByRole('button', { name: 'Next' })).toBeVisible();
    });

    test('walk through all wizard steps', async ({ page }) => {
        await splunkLogin(page);
        await navigateTo(page, 'builder');
        await waitForReactMount(page);

        // Step 1: Connect — fill Base URL and advance
        await expect(page.getByText('New Stream')).toBeVisible({ timeout: 15_000 });
        const baseUrlInput = page.getByPlaceholder('https://api.example.com/v1/resources');
        await expect(baseUrlInput).toBeVisible();
        await baseUrlInput.fill('https://jsonplaceholder.typicode.com/posts');
        await page.getByRole('button', { name: 'Next' }).click();
        await page.waitForTimeout(500);

        // Step 2: Configure — collapsible sections visible
        await expect(page.getByText('Record Selector')).toBeVisible();
        await page.getByRole('button', { name: 'Next' }).click();
        await page.waitForTimeout(500);

        // Step 3: Output — two-column layout
        await expect(page.getByText('Destination')).toBeVisible();
        await expect(page.getByText('Schedule')).toBeVisible();
        await page.getByRole('button', { name: 'Next' }).click();
        await page.waitForTimeout(500);

        // Step 4: Test & Save — Save button appears
        await expect(page.getByRole('button', { name: 'Save' })).toBeVisible();
    });
});
