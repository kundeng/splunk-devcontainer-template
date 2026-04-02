/**
 * URC App — Create Stream E2E (wizard flow)
 */

import { test, expect } from '@playwright/test';
import { splunkLogin, navigateTo, waitForReactMount, deleteAllStreams } from './helpers';

test.describe('Create Stream Flow', () => {
    test.beforeEach(async ({ page }) => {
        await splunkLogin(page);
        await deleteAllStreams(page);
    });

    test('create JSONPlaceholder stream via wizard', async ({ page }) => {
        await navigateTo(page, 'builder');
        await waitForReactMount(page);
        await expect(page.getByText('New Stream')).toBeVisible({ timeout: 15_000 });

        // Step 1: Connect — fill Base URL
        const baseUrlInput = page.getByPlaceholder('https://api.example.com/v1/resources');
        await expect(baseUrlInput).toBeVisible();
        await baseUrlInput.fill('https://jsonplaceholder.typicode.com/posts');
        await page.getByRole('button', { name: 'Next' }).click();
        await page.waitForTimeout(500);

        // Step 2: Configure — just advance
        await page.getByRole('button', { name: 'Next' }).click();
        await page.waitForTimeout(500);

        // Step 3: Output — just advance
        await page.getByRole('button', { name: 'Next' }).click();
        await page.waitForTimeout(500);

        // Step 4: Save
        const saveButton = page.getByRole('button', { name: 'Save' });
        await expect(saveButton).toBeVisible();
        await saveButton.click();
        await page.waitForTimeout(5000);

        // Should redirect to streams dashboard
        const currentUrl = page.url();
        console.log('After save URL:', currentUrl);
        await page.screenshot({ path: 'tests/e2e/playwright/screenshots/save-after-wizard.png' });
    });
});
