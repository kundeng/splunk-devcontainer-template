import { Page, expect } from '@playwright/test';

const SPLUNK_USER = process.env.SPLUNK_USER || 'admin';
const SPLUNK_PASS = process.env.SPLUNK_PASSWORD || 'admin123';

/**
 * Log in to Splunk and navigate to the URC app.
 * Saves session cookies so subsequent navigations stay authenticated.
 */
export async function splunkLogin(page: Page): Promise<void> {
    await page.goto('/en-US/account/login');
    await page.fill('input[name="username"]', SPLUNK_USER);
    await page.fill('input[name="password"]', SPLUNK_PASS);
    await page.click('input[type="submit"]');
    // Wait for redirect to land on a real page (not login)
    await page.waitForURL((url) => !url.pathname.includes('/account/login'), {
        timeout: 15_000,
    });
}

/**
 * Navigate to a URC app view and wait for the page JS to load.
 */
export async function navigateTo(page: Page, view: string): Promise<void> {
    await page.goto(`/en-US/app/urc_app/${view}`);
    await page.waitForLoadState('networkidle');
}

/**
 * Wait for the React app to mount (Splunk's layout wrapper renders a div#app).
 */
export async function waitForReactMount(page: Page): Promise<void> {
    // Splunk React pages mount into a container — wait for meaningful content
    await page.waitForTimeout(2000);
    await page.waitForLoadState('networkidle');
}

/**
 * Delete all URC inputs (streams) via REST API to clean test state.
 */
export async function deleteAllStreams(page: Page): Promise<void> {
    // Get CSRF token from cookies
    const cookies = await page.context().cookies();
    const csrfCookie = cookies.find((c) => c.name.startsWith('splunkweb_csrf_token'));
    const formKey = csrfCookie?.value || '';

    const headers: Record<string, string> = {
        'X-Splunk-Form-Key': formKey,
        'X-Requested-With': 'XMLHttpRequest',
    };

    // Use Splunk REST to list and delete all inputs
    const response = await page.request.get(
        `/en-US/splunkd/__raw/services/data/inputs/urc_app_input?output_mode=json&count=0`,
        { headers }
    );
    if (response.ok()) {
        const data = await response.json();
        const entries = data.entry || [];
        for (const entry of entries) {
            const name = entry.name;
            await page.request.delete(
                `/en-US/splunkd/__raw/services/data/inputs/urc_app_input/${encodeURIComponent(name)}?output_mode=json`,
                { headers }
            );
        }
    }
}
