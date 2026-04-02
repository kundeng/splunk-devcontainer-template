import { test, expect } from '@playwright/test';

const SPLUNK_USER = process.env.SPLUNK_USER || 'admin';
const SPLUNK_PASS = process.env.SPLUNK_PASSWORD || 'admin123';

test('UCC pages with network capture', async ({ page }) => {
    // Login
    await page.goto('/en-US/account/login');
    await page.fill('input[name="username"]', SPLUNK_USER);
    await page.fill('input[name="password"]', SPLUNK_PASS);
    await page.click('input[type="submit"]');
    await page.waitForURL((url) => !url.pathname.includes('/account/login'), { timeout: 15_000 });

    // Capture network errors
    page.on('response', async (resp) => {
        if (resp.status() >= 400 && resp.url().includes('splunkd')) {
            const body = await resp.text().catch(() => '');
            console.log(`[${resp.status()}] ${resp.url()}`);
            if (body.length < 500) console.log(`  Body: ${body}`);
        }
    });

    // Inputs page
    console.log('\n=== INPUTS PAGE ===');
    await page.goto('/en-US/app/urc_app/inputs');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(5000);
    await page.screenshot({ path: 'tests/e2e/playwright/screenshots/verify-ucc-inputs.png', fullPage: true });

    // Configuration page
    console.log('\n=== CONFIGURATION PAGE ===');
    await page.goto('/en-US/app/urc_app/configuration');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(5000);
    await page.screenshot({ path: 'tests/e2e/playwright/screenshots/verify-ucc-config.png', fullPage: true });
});
