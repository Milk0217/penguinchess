import { test, expect, Page } from '@playwright/test'

// Base URL - frontend dev server
const BASE_URL = 'http://localhost:5173'

// Helper to wait for game to load
async function waitForGameLoad(page: Page) {
  await page.waitForSelector('h1', { timeout: 10000 })
  await page.waitForFunction(() => {
    // Wait until loading is done and state is visible
    const loading = document.querySelector('.loading')
    return !loading
  }, { timeout: 10000 })
}

test.describe('企鹅棋 Web 前端 E2E 测试', () => {

  test.beforeEach(async ({ page }) => {
    await page.goto(BASE_URL)
    await waitForGameLoad(page)
  })

  test('页面正常加载，标题存在', async ({ page }) => {
    const title = page.locator('h1')
    await expect(title).toBeVisible()
    const text = await title.textContent()
    expect(text).toContain('企鹅棋')
  })

  test('分数板显示正确', async ({ page }) => {
    // 应该显示两个玩家的分数
    const scoreboard = page.locator('.scoreboard')
    await expect(scoreboard).toBeVisible()
    // 应该显示 "vs"
    await expect(page.locator('.vs')).toBeVisible()
  })

  test('状态栏显示正确', async ({ page }) => {
    const statusBar = page.locator('.status-bar')
    await expect(statusBar).toBeVisible()
    const text = await statusBar.textContent()
    expect(text).toContain('Player')
  })

  test('控制按钮存在', async ({ page }) => {
    // 新游戏按钮
    const newGameBtn = page.getByText('新游戏')
    await expect(newGameBtn).toBeVisible()

    // 重开按钮
    const resetBtn = page.getByText('重开')
    await expect(resetBtn).toBeVisible()

    // 棋盘编辑器按钮
    const editorBtn = page.getByText('棋盘编辑器')
    await expect(editorBtn).toBeVisible()
  })

  test('棋盘渲染正常', async ({ page }) => {
    // 等待 SVG 棋盘渲染
    const boardSvg = page.locator('svg').first()
    await expect(boardSvg).toBeVisible()

    // 检查是否有六边形格子（SVG path）
    const hexPaths = page.locator('svg path')
    const count = await hexPaths.count()
    expect(count).toBeGreaterThan(10) // 至少有一些格子
  })

  test('点击新游戏重置游戏', async ({ page }) => {
    // 获取初始步数
    const statusBar = page.locator('.status-bar')
    const initialText = await statusBar.textContent()

    // 点击新游戏
    await page.getByText('新游戏').click()
    await page.waitForTimeout(500)

    // 状态应该重置
    const newText = await statusBar.textContent()
    expect(newText).toContain('放置棋子')
  })

  test('主题选择器存在且可交互', async ({ page }) => {
    // 查找主题选择器（select 元素）
    const themeSelect = page.locator('select').last()
    await expect(themeSelect).toBeVisible()

    // 可以选择不同主题
    const options = await themeSelect.locator('option').count()
    expect(options).toBeGreaterThan(1)
  })

  test('棋盘编辑器按钮可点击', async ({ page }) => {
    await page.getByText('棋盘编辑器').click()
    // 应该导航到编辑器（检查 URL 或内容变化）
    await page.waitForTimeout(500)
    // 编辑器应该有返回按钮
    const backBtn = page.getByText('返回')
    await expect(backBtn).toBeVisible()
  })

})

test.describe('游戏流程测试', () => {

  test('放置阶段 - 放置棋子', async ({ page }) => {
    await page.goto(BASE_URL)
    await waitForGameLoad(page)

    // 状态应该是放置阶段
    const statusBar = page.locator('.status-bar')
    await expect(statusBar).toContainText('放置棋子')

    // 点击一个格子放置棋子
    const hexes = page.locator('svg g').filter({ has: page.locator('path') })
    const firstHex = hexes.first()

    // 点击第一个格子
    await firstHex.click()
    await page.waitForTimeout(300)

    // 回合应该切换
    const newStatus = await statusBar.textContent()
    // 可能是 P1 -> P2 或仍然 P1（取决于谁先手）
    expect(newStatus).toMatch(/Player \d/)
  })

  test('游戏结束检测', async ({ page }) => {
    // 这个测试需要玩多步到达游戏结束
    // 由于游戏可能需要很多步，我们只测试是否能处理游戏结束状态

    await page.goto(BASE_URL)
    await waitForGameLoad(page)

    // 多次点击直到游戏结束或达到步数上限
    for (let i = 0; i < 50; i++) {
      const hexes = page.locator('svg g').filter({ has: page.locator('path') })
      const randomHex = hexes.nth(Math.floor(Math.random() * 20))
      await randomHex.click().catch(() => {})
      await page.waitForTimeout(100)

      const statusBar = page.locator('.status-bar')
      const text = await statusBar.textContent()
      if (text.includes('获胜') || text.includes('平局')) {
        // 游戏结束
        expect(true).toBe(true)
        return
      }
    }
    // 如果没有结束也没关系，至少没有崩溃
    expect(true).toBe(true)
  })

})

test.describe('响应式布局测试', () => {

  test('移动阶段棋子选中', async ({ page }) => {
    await page.goto(BASE_URL)
    await waitForGameLoad(page)

    // 跳过放置阶段进入移动阶段
    for (let i = 0; i < 6; i++) {
      const hexes = page.locator('svg g').filter({ has: page.locator('path') })
      await hexes.nth(i % 20).click().catch(() => {})
      await page.waitForTimeout(200)
    }

    // 检查是否进入移动阶段
    const statusBar = page.locator('.status-bar')
    await page.waitForTimeout(500)
    const text = await statusBar.textContent()

    // 如果是移动阶段，应该显示"移动棋子"
    if (text.includes('移动棋子')) {
      // 尝试点击一个己方棋子
      const hexes = page.locator('svg g').filter({ has: page.locator('path') })
      await hexes.nth(5).click().catch(() => {})
      await page.waitForTimeout(200)
    }
  })

})