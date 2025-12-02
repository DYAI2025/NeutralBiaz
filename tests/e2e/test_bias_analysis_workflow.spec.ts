/**
 * End-to-end workflow tests for BiazNeutralize AI system
 * Tests complete user journeys from bias analysis to neutralization
 */
import { test, expect, Page, BrowserContext } from '@playwright/test';

// Test configuration
const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';
const API_URL = process.env.API_URL || 'http://localhost:8000';

// Test data
const TEST_TEXTS = {
  highBias: "This research clearly proves that our original hypothesis was completely correct and any studies contradicting our findings are obviously flawed.",
  mediumBias: "Based on the initial assessment, the evidence suggests that this approach is reasonable.",
  lowBias: "The study examined multiple variables and found mixed results that require further investigation.",
  edgeCase: "Short.",
  longText: "This is an extensive piece of text designed to test the system's ability to handle longer content submissions. ".repeat(50),
  specialChars: "Text with émojis 🚀 and spëciál charâcters! 你好世界",
  cultural: "Individual achievement should always take priority over group harmony and collective decision-making processes."
};

class BiasAnalysisPage {
  constructor(private page: Page) {}

  async navigate() {
    await this.page.goto(BASE_URL);
    await this.page.waitForLoadState('networkidle');
  }

  async enterText(text: string) {
    const textInput = this.page.locator('[data-testid="text-input"], textarea[placeholder*="analyze"]');
    await textInput.clear();
    await textInput.fill(text);
  }

  async selectCulturalContext(context: string) {
    const select = this.page.locator('[data-testid="cultural-context-select"], select');
    await select.selectOption(context);
  }

  async clickAnalyzeButton() {
    const button = this.page.locator('[data-testid="analyze-button"], button:has-text("Analyze")');
    await button.click();
  }

  async waitForResults() {
    await this.page.waitForSelector('[data-testid="analysis-result"], [aria-label="Analysis Results"]', {
      timeout: 30000
    });
  }

  async getOverallScore(): Promise<number> {
    const scoreElement = this.page.locator('[data-testid="overall-score"]').first();
    const scoreText = await scoreElement.textContent();
    const match = scoreText?.match(/(\d+\.?\d*)%/);
    return match ? parseFloat(match[1]) / 100 : 0;
  }

  async getConfidenceScore(): Promise<number> {
    const confidenceElement = this.page.locator('[data-testid="confidence-score"]').first();
    const confidenceText = await confidenceElement.textContent();
    const match = confidenceText?.match(/(\d+\.?\d*)%/);
    return match ? parseFloat(match[1]) / 100 : 0;
  }

  async getDetectedBiases(): Promise<string[]> {
    const biasElements = this.page.locator('[data-testid="detected-biases"] .bias-type, [data-testid="detected-biases"] li');
    const biases: string[] = [];
    const count = await biasElements.count();

    for (let i = 0; i < count; i++) {
      const biasText = await biasElements.nth(i).textContent();
      if (biasText) {
        biases.push(biasText.trim());
      }
    }

    return biases;
  }

  async getSuggestions(): Promise<string[]> {
    const suggestionElements = this.page.locator('[data-testid="suggestions-list"] li, .suggestions li');
    const suggestions: string[] = [];
    const count = await suggestionElements.count();

    for (let i = 0; i < count; i++) {
      const suggestionText = await suggestionElements.nth(i).textContent();
      if (suggestionText) {
        suggestions.push(suggestionText.trim());
      }
    }

    return suggestions;
  }

  async clickNeutralizeButton() {
    const button = this.page.locator('button:has-text("Neutralize"), [data-testid="neutralize-button"]');
    await button.click();
  }

  async getNeutralizedText(): Promise<string> {
    const neutralizedElement = this.page.locator('[data-testid="neutralized-text"], .neutralized-output');
    return await neutralizedElement.textContent() || '';
  }
}

class DashboardPage {
  constructor(private page: Page) {}

  async navigate() {
    await this.page.goto(`${BASE_URL}/dashboard`);
    await this.page.waitForLoadState('networkidle');
  }

  async clickNewAnalysis() {
    const button = this.page.locator('[data-testid="new-analysis-button"], button:has-text("New Analysis")');
    await button.click();
  }

  async getAnalysisHistory(): Promise<any[]> {
    const rows = this.page.locator('[data-testid="analyses-table"] tbody tr, table tbody tr');
    const history: any[] = [];
    const count = await rows.count();

    for (let i = 0; i < count; i++) {
      const row = rows.nth(i);
      const cells = row.locator('td');

      if (await cells.count() >= 3) {
        const text = await cells.nth(0).textContent();
        const score = await cells.nth(1).textContent();
        const date = await cells.nth(2).textContent();

        history.push({
          text: text?.trim(),
          score: score?.trim(),
          date: date?.trim()
        });
      }
    }

    return history;
  }

  async viewAnalysisDetails(index: number) {
    const button = this.page.locator(`[data-testid="view-details-${index}"], tbody tr:nth-child(${index + 1}) button:has-text("View")`);
    await button.click();
  }

  async waitForStatsUpdate() {
    // Wait for statistics to update after new analysis
    await this.page.waitForFunction(() => {
      const totalElement = document.querySelector('[data-testid="stat-card-analyses"] .stat-value, .stats .total-analyses');
      return totalElement && parseInt(totalElement.textContent || '0') > 0;
    }, { timeout: 10000 });
  }
}

// Test suite
test.describe('Bias Analysis Workflow', () => {
  let biasAnalysisPage: BiasAnalysisPage;
  let dashboardPage: DashboardPage;

  test.beforeEach(async ({ page }) => {
    biasAnalysisPage = new BiasAnalysisPage(page);
    dashboardPage = new DashboardPage(page);
  });

  test.describe('Core Analysis Functionality', () => {
    test('should complete basic bias analysis workflow', async ({ page }) => {
      await biasAnalysisPage.navigate();

      // Enter text with obvious bias
      await biasAnalysisPage.enterText(TEST_TEXTS.highBias);

      // Start analysis
      await biasAnalysisPage.clickAnalyzeButton();

      // Wait for results
      await biasAnalysisPage.waitForResults();

      // Verify results
      const overallScore = await biasAnalysisPage.getOverallScore();
      expect(overallScore).toBeGreaterThan(0.5); // High bias text should have high score

      const confidence = await biasAnalysisPage.getConfidenceScore();
      expect(confidence).toBeGreaterThan(0.6); // Should be confident

      const detectedBiases = await biasAnalysisPage.getDetectedBiases();
      expect(detectedBiases.length).toBeGreaterThan(0); // Should detect at least one bias

      // Verify suggestions are provided for high bias
      const suggestions = await biasAnalysisPage.getSuggestions();
      expect(suggestions.length).toBeGreaterThan(0);
    });

    test('should handle low bias text correctly', async ({ page }) => {
      await biasAnalysisPage.navigate();

      await biasAnalysisPage.enterText(TEST_TEXTS.lowBias);
      await biasAnalysisPage.clickAnalyzeButton();
      await biasAnalysisPage.waitForResults();

      const overallScore = await biasAnalysisPage.getOverallScore();
      expect(overallScore).toBeLessThan(0.4); // Low bias text should have low score

      const confidence = await biasAnalysisPage.getConfidenceScore();
      expect(confidence).toBeGreaterThan(0.5); // Should still be reasonably confident
    });

    test('should process different text lengths', async ({ page }) => {
      const testCases = [
        { text: TEST_TEXTS.edgeCase, name: 'very short text' },
        { text: TEST_TEXTS.longText, name: 'very long text' }
      ];

      for (const testCase of testCases) {
        await test.step(`Processing ${testCase.name}`, async () => {
          await biasAnalysisPage.navigate();
          await biasAnalysisPage.enterText(testCase.text);
          await biasAnalysisPage.clickAnalyzeButton();

          // Should complete within reasonable time
          await biasAnalysisPage.waitForResults();

          // Should return valid results
          const overallScore = await biasAnalysisPage.getOverallScore();
          expect(overallScore).toBeGreaterThanOrEqual(0);
          expect(overallScore).toBeLessThanOrEqual(1);
        });
      }
    });

    test('should handle special characters and unicode', async ({ page }) => {
      await biasAnalysisPage.navigate();

      await biasAnalysisPage.enterText(TEST_TEXTS.specialChars);
      await biasAnalysisPage.clickAnalyzeButton();
      await biasAnalysisPage.waitForResults();

      // Should process without errors
      const overallScore = await biasAnalysisPage.getOverallScore();
      expect(overallScore).toBeGreaterThanOrEqual(0);
    });
  });

  test.describe('Cultural Context Testing', () => {
    test('should adapt analysis based on cultural context', async ({ page }) => {
      const culturalContexts = ['en-US', 'ja-JP', 'de-DE'];
      const results: { [key: string]: number } = {};

      for (const context of culturalContexts) {
        await test.step(`Testing cultural context: ${context}`, async () => {
          await biasAnalysisPage.navigate();
          await biasAnalysisPage.enterText(TEST_TEXTS.cultural);
          await biasAnalysisPage.selectCulturalContext(context);
          await biasAnalysisPage.clickAnalyzeButton();
          await biasAnalysisPage.waitForResults();

          const score = await biasAnalysisPage.getOverallScore();
          results[context] = score;
        });
      }

      // Results should vary between cultures (at least 0.1 difference)
      const scores = Object.values(results);
      const maxScore = Math.max(...scores);
      const minScore = Math.min(...scores);
      expect(maxScore - minScore).toBeGreaterThan(0.1);
    });
  });

  test.describe('Text Neutralization Workflow', () => {
    test('should neutralize biased text', async ({ page }) => {
      await biasAnalysisPage.navigate();

      // Analyze biased text
      await biasAnalysisPage.enterText(TEST_TEXTS.highBias);
      await biasAnalysisPage.clickAnalyzeButton();
      await biasAnalysisPage.waitForResults();

      const originalScore = await biasAnalysisPage.getOverallScore();
      expect(originalScore).toBeGreaterThan(0.5);

      // Neutralize the text
      await biasAnalysisPage.clickNeutralizeButton();

      // Wait for neutralization results
      await page.waitForSelector('[data-testid="neutralized-text"], .neutralized-output', {
        timeout: 15000
      });

      const neutralizedText = await biasAnalysisPage.getNeutralizedText();
      expect(neutralizedText).toBeTruthy();
      expect(neutralizedText).not.toBe(TEST_TEXTS.highBias);

      // Neutralized text should be different from original
      expect(neutralizedText.length).toBeGreaterThan(0);
    });
  });

  test.describe('Dashboard Integration', () => {
    test('should show analysis history in dashboard', async ({ page }) => {
      // Perform an analysis
      await biasAnalysisPage.navigate();
      await biasAnalysisPage.enterText(TEST_TEXTS.mediumBias);
      await biasAnalysisPage.clickAnalyzeButton();
      await biasAnalysisPage.waitForResults();

      // Navigate to dashboard
      await dashboardPage.navigate();

      // Check if analysis appears in history
      await dashboardPage.waitForStatsUpdate();

      const history = await dashboardPage.getAnalysisHistory();
      expect(history.length).toBeGreaterThan(0);

      // Verify the analysis data
      const latestAnalysis = history[0];
      expect(latestAnalysis.text).toContain(TEST_TEXTS.mediumBias.substring(0, 20));
      expect(latestAnalysis.score).toMatch(/\d+\.?\d*%/);
    });

    test('should allow starting new analysis from dashboard', async ({ page }) => {
      await dashboardPage.navigate();
      await dashboardPage.clickNewAnalysis();

      // Should navigate to analysis page
      await expect(page).toHaveURL(/.*\/(analyze|bias-analysis)/);

      // Should be able to enter text immediately
      const textInput = page.locator('textarea, [data-testid="text-input"]');
      await expect(textInput).toBeVisible();
    });

    test('should display updated statistics after analysis', async ({ page }) => {
      // Get initial stats
      await dashboardPage.navigate();
      const initialHistory = await dashboardPage.getAnalysisHistory();
      const initialCount = initialHistory.length;

      // Perform new analysis
      await dashboardPage.clickNewAnalysis();
      await biasAnalysisPage.enterText(TEST_TEXTS.highBias);
      await biasAnalysisPage.clickAnalyzeButton();
      await biasAnalysisPage.waitForResults();

      // Return to dashboard and check updated stats
      await dashboardPage.navigate();
      await dashboardPage.waitForStatsUpdate();

      const updatedHistory = await dashboardPage.getAnalysisHistory();
      expect(updatedHistory.length).toBe(initialCount + 1);
    });
  });

  test.describe('Error Handling and Edge Cases', () => {
    test('should handle empty text input', async ({ page }) => {
      await biasAnalysisPage.navigate();

      // Try to analyze empty text
      await biasAnalysisPage.enterText('');

      // Button should be disabled
      const analyzeButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")');
      await expect(analyzeButton).toBeDisabled();
    });

    test('should handle very long processing times gracefully', async ({ page }) => {
      await biasAnalysisPage.navigate();

      await biasAnalysisPage.enterText(TEST_TEXTS.longText);
      await biasAnalysisPage.clickAnalyzeButton();

      // Should show loading state
      const loadingIndicator = page.locator('[data-testid="loading-indicator"], .loading, .spinner');
      await expect(loadingIndicator).toBeVisible();

      // Should eventually complete
      await biasAnalysisPage.waitForResults();

      // Loading should be hidden
      await expect(loadingIndicator).not.toBeVisible();
    });

    test('should handle network errors gracefully', async ({ page, context }) => {
      // Intercept API calls to simulate network error
      await page.route(`${API_URL}/**`, route => route.abort());

      await biasAnalysisPage.navigate();
      await biasAnalysisPage.enterText(TEST_TEXTS.mediumBias);
      await biasAnalysisPage.clickAnalyzeButton();

      // Should show error state or retry mechanism
      await page.waitForSelector('.error, [role="alert"], .error-message', {
        timeout: 10000
      });

      const errorMessage = page.locator('.error, [role="alert"], .error-message');
      await expect(errorMessage).toBeVisible();
    });
  });

  test.describe('Performance and Usability', () => {
    test('should meet response time requirements', async ({ page }) => {
      await biasAnalysisPage.navigate();

      await biasAnalysisPage.enterText(TEST_TEXTS.mediumBias);

      const startTime = Date.now();
      await biasAnalysisPage.clickAnalyzeButton();
      await biasAnalysisPage.waitForResults();
      const endTime = Date.now();

      const responseTime = (endTime - startTime) / 1000;

      // Should complete within 5 seconds (allowing for network latency in tests)
      expect(responseTime).toBeLessThan(5);
    });

    test('should maintain responsiveness during analysis', async ({ page }) => {
      await biasAnalysisPage.navigate();

      await biasAnalysisPage.enterText(TEST_TEXTS.longText);
      await biasAnalysisPage.clickAnalyzeButton();

      // UI should remain responsive during processing
      const textInput = page.locator('textarea, [data-testid="text-input"]');
      await textInput.click();
      await textInput.type(' additional text');

      // Should be able to interact with other elements
      const culturalSelect = page.locator('[data-testid="cultural-context-select"], select');
      await culturalSelect.click();
    });
  });

  test.describe('Accessibility Compliance', () => {
    test('should be keyboard navigable', async ({ page }) => {
      await biasAnalysisPage.navigate();

      // Should be able to navigate with Tab key
      await page.keyboard.press('Tab'); // Text input
      await page.keyboard.type(TEST_TEXTS.mediumBias);

      await page.keyboard.press('Tab'); // Cultural context select
      await page.keyboard.press('Tab'); // Analyze button
      await page.keyboard.press('Enter'); // Activate button

      // Should start analysis
      await biasAnalysisPage.waitForResults();

      const overallScore = await biasAnalysisPage.getOverallScore();
      expect(overallScore).toBeGreaterThanOrEqual(0);
    });

    test('should have proper ARIA labels and roles', async ({ page }) => {
      await biasAnalysisPage.navigate();

      // Check for accessibility attributes
      const textInput = page.locator('textarea, [data-testid="text-input"]');
      await expect(textInput).toHaveAttribute('aria-label');

      const analyzeButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")');
      await expect(analyzeButton).toHaveAttribute('type', 'button');

      // Perform analysis to test results accessibility
      await biasAnalysisPage.enterText(TEST_TEXTS.mediumBias);
      await biasAnalysisPage.clickAnalyzeButton();
      await biasAnalysisPage.waitForResults();

      const resultsRegion = page.locator('[role="region"], [data-testid="analysis-result"]');
      await expect(resultsRegion).toBeVisible();
    });
  });

  test.describe('Cross-browser Compatibility', () => {
    ['chromium', 'firefox', 'webkit'].forEach(browserName => {
      test(`should work in ${browserName}`, async ({ browser }) => {
        test.skip(browserName !== 'chromium', `Skipping ${browserName} for faster testing`);

        const context = await browser.newContext();
        const page = await context.newPage();
        const analysisPage = new BiasAnalysisPage(page);

        await analysisPage.navigate();
        await analysisPage.enterText(TEST_TEXTS.mediumBias);
        await analysisPage.clickAnalyzeButton();
        await analysisPage.waitForResults();

        const overallScore = await analysisPage.getOverallScore();
        expect(overallScore).toBeGreaterThanOrEqual(0);
        expect(overallScore).toBeLessThanOrEqual(1);

        await context.close();
      });
    });
  });

  test.describe('Mobile Responsiveness', () => {
    test('should work on mobile viewport', async ({ page }) => {
      // Set mobile viewport
      await page.setViewportSize({ width: 375, height: 667 });

      await biasAnalysisPage.navigate();

      // Should be usable on mobile
      const textInput = page.locator('textarea, [data-testid="text-input"]');
      await expect(textInput).toBeVisible();

      await biasAnalysisPage.enterText(TEST_TEXTS.mediumBias);
      await biasAnalysisPage.clickAnalyzeButton();
      await biasAnalysisPage.waitForResults();

      // Results should be displayed properly
      const overallScore = await biasAnalysisPage.getOverallScore();
      expect(overallScore).toBeGreaterThanOrEqual(0);

      // Elements should not overflow
      const body = page.locator('body');
      const bodyWidth = await body.evaluate(el => el.scrollWidth);
      expect(bodyWidth).toBeLessThanOrEqual(375 + 20); // Allow for small scrollbar
    });
  });
});

// Test configuration
test.describe.configure({ mode: 'parallel', timeout: 60000 });

test.afterEach(async ({ page }, testInfo) => {
  if (testInfo.status !== testInfo.expectedStatus) {
    // Capture screenshot on failure
    const screenshot = await page.screenshot({ fullPage: true });
    await testInfo.attach('screenshot', { body: screenshot, contentType: 'image/png' });
  }
});

test.afterAll(async () => {
  // Cleanup any test data if needed
  console.log('E2E tests completed');
});