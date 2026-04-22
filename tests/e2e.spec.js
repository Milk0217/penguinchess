import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:5000';

test.describe('企鹅棋 E2E 测试', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(BASE_URL);
    });

    test('页面正常加载，标题存在', async ({ page }) => {
        await expect(page.locator('h1')).toBeVisible();
        const title = await page.locator('h1').textContent();
        expect(title).toBeTruthy();
    });

    test('棋盘正常渲染', async ({ page }) => {
        await page.waitForSelector('#board');
        const hexCount = await page.locator('.hex').count();
        expect(hexCount).toBeGreaterThan(0);
    });

    test('控制按钮存在', async ({ page }) => {
        await expect(page.locator('#startBtn')).toBeVisible();
        await expect(page.locator('#resetBtn')).toBeVisible();
        await expect(page.locator('#toggle-coords-btn')).toBeVisible();
    });

    test('点击开始按钮，棋子被放置', async ({ page }) => {
        await page.click('#startBtn');
        await page.waitForTimeout(500);
        const pieceCount = await page.locator('.piece').count();
        expect(pieceCount).toBeGreaterThan(0);
    });

    test('坐标切换按钮正常工作', async ({ page }) => {
        const firstHex = page.locator('.hex').first();
        const initialText = await firstHex.textContent();

        await page.click('#toggle-coords-btn');
        const toggledText = await firstHex.textContent();

        expect(toggledText).not.toBe(initialText);
    });
});
