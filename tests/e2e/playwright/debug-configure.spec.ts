import { test } from '@playwright/test';

test.use({ viewport: { width: 1280, height: 1200 } });

test('screenshot configure step expanded', async ({ page }) => {
    await page.goto('/en-US/account/login');
    await page.fill('input[name="username"]', 'admin');
    await page.fill('input[name="password"]', 'admin123');
    await page.click('input[type="submit"]');
    await page.waitForURL((url) => !url.pathname.includes('/account/login'), { timeout: 15_000 });

    await page.goto('/en-US/app/urc_app/builder');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);

    // Fill URL and go to Configure step
    const urlInput = page.getByPlaceholder('https://api.example.com/v1/resources');
    await urlInput.fill('https://jsonplaceholder.typicode.com/posts');
    await page.getByRole('button', { name: 'Next' }).click();
    await page.waitForTimeout(1000);

    // Screenshot collapsed
    await page.screenshot({ path: 'tests/e2e/playwright/screenshots/configure-collapsed.png', fullPage: true });

    // Expand Record Selector
    const recordSelector = page.getByText('Record Selector');
    await recordSelector.click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: 'tests/e2e/playwright/screenshots/configure-record-selector.png', fullPage: true });

    // Select DpathExtractor type
    const typeSelect = page.locator('[data-test="select"]').first();
    if (await typeSelect.isVisible({ timeout: 2000 }).catch(() => false)) {
        await typeSelect.click();
        await page.waitForTimeout(500);
        await page.screenshot({ path: 'tests/e2e/playwright/screenshots/configure-type-dropdown.png', fullPage: true });
        // Select DpathExtractor
        const dpathOption = page.locator('[data-test="option"]').filter({ hasText: 'Dpath' }).first();
        if (await dpathOption.isVisible({ timeout: 2000 }).catch(() => false)) {
            await dpathOption.click();
            await page.waitForTimeout(500);
        }
    }
    await page.screenshot({ path: 'tests/e2e/playwright/screenshots/configure-dpath-selected.png', fullPage: true });

    // Expand Pagination
    const pagination = page.getByText('Pagination');
    await pagination.click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: 'tests/e2e/playwright/screenshots/configure-pagination.png', fullPage: true });
});
