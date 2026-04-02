import { test } from '@playwright/test';

test.use({ viewport: { width: 1280, height: 900 } });

test('debug builder page', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));
    page.on('console', (msg) => {
        if (msg.type() === 'error') errors.push(msg.text());
    });

    await page.goto('/en-US/account/login');
    await page.fill('input[name="username"]', 'admin');
    await page.fill('input[name="password"]', 'admin123');
    await page.click('input[type="submit"]');
    await page.waitForURL((url) => !url.pathname.includes('/account/login'), { timeout: 15_000 });

    await page.goto('/en-US/app/urc_app/builder');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(5000);

    console.log('=== ERRORS ===');
    for (const e of errors) console.log(e);
    console.log(`Total: ${errors.length} errors`);

    await page.screenshot({ path: 'tests/e2e/playwright/screenshots/debug-builder.png', fullPage: true });
});
