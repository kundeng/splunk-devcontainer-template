/**
 * Fresh verification — screenshot every page and wizard step.
 */
import { test } from '@playwright/test';

const SPLUNK_USER = process.env.SPLUNK_USER || 'admin';
const SPLUNK_PASS = process.env.SPLUNK_PASSWORD || 'admin123';

test('screenshot everything', async ({ page }) => {
    // Login
    await page.goto('/en-US/account/login');
    await page.fill('input[name="username"]', SPLUNK_USER);
    await page.fill('input[name="password"]', SPLUNK_PASS);
    await page.click('input[type="submit"]');
    await page.waitForURL((url) => !url.pathname.includes('/account/login'), { timeout: 15_000 });

    // Streams dashboard
    await page.goto('/en-US/app/urc_app/streams');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);
    await page.screenshot({ path: 'tests/e2e/playwright/screenshots/redesign-01-dashboard.png', fullPage: true });

    // Builder — Step 1: Connect
    await page.goto('/en-US/app/urc_app/builder');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);
    await page.screenshot({ path: 'tests/e2e/playwright/screenshots/redesign-02-step1-connect.png', fullPage: true });

    // Step 2: Configure (click Next)
    const nextButton = page.getByRole('button', { name: 'Next' });
    // Fill base URL first so validation passes
    const urlInput = page.getByPlaceholder('https://api.example.com/v1/resources');
    if (await urlInput.isVisible({ timeout: 2000 }).catch(() => false)) {
        await urlInput.fill('https://jsonplaceholder.typicode.com/posts');
    }
    await nextButton.click();
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'tests/e2e/playwright/screenshots/redesign-03-step2-configure.png', fullPage: true });

    // Step 3: Output (click Next)
    await nextButton.click();
    await page.waitForTimeout(2000);
    await page.screenshot({ path: 'tests/e2e/playwright/screenshots/redesign-04-step3-output.png', fullPage: true });

    // Step 4: Test & Save (click Next)
    await nextButton.click();
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'tests/e2e/playwright/screenshots/redesign-05-step4-test-save.png', fullPage: true });

    // Configuration page
    await page.goto('/en-US/app/urc_app/configuration');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);
    await page.screenshot({ path: 'tests/e2e/playwright/screenshots/redesign-06-configuration.png', fullPage: true });
});
