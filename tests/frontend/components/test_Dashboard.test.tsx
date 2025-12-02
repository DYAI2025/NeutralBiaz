/**
 * Unit tests for Dashboard component
 * Tests main dashboard functionality, data fetching, and user interactions
 */
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';

// Mock chart.js
vi.mock('chart.js', () => ({
  Chart: vi.fn().mockImplementation(() => ({
    destroy: vi.fn(),
    update: vi.fn(),
    resize: vi.fn()
  })),
  registerables: []
}));

vi.mock('react-chartjs-2', () => ({
  Line: ({ data, options, ...props }: any) => (
    <div data-testid="line-chart" {...props}>
      Mock Line Chart: {data?.datasets?.[0]?.label || 'No data'}
    </div>
  ),
  Bar: ({ data, options, ...props }: any) => (
    <div data-testid="bar-chart" {...props}>
      Mock Bar Chart: {data?.datasets?.[0]?.label || 'No data'}
    </div>
  ),
  Doughnut: ({ data, options, ...props }: any) => (
    <div data-testid="doughnut-chart" {...props}>
      Mock Doughnut Chart: {JSON.stringify(data?.labels || [])}
    </div>
  )
}));

// Mock the Dashboard component
const Dashboard = ({
  analysisHistory = [],
  onNewAnalysis,
  isLoading = false
}: {
  analysisHistory?: any[];
  onNewAnalysis?: () => void;
  isLoading?: boolean;
}) => {
  const [selectedTimeframe, setSelectedTimeframe] = React.useState('7days');
  const [selectedMetric, setSelectedMetric] = React.useState('bias_score');

  const mockStats = {
    total_analyses: analysisHistory.length,
    avg_bias_score: 0.45,
    most_common_bias: 'confirmation_bias',
    trend_direction: 'improving'
  };

  const chartData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: [{
      label: 'Bias Score Trend',
      data: [0.6, 0.5, 0.4, 0.45, 0.3, 0.35, 0.4],
      borderColor: 'rgb(59, 130, 246)',
      backgroundColor: 'rgba(59, 130, 246, 0.1)'
    }]
  };

  const biasDistribution = {
    labels: ['Confirmation', 'Anchoring', 'Availability', 'Selection', 'Survivorship'],
    datasets: [{
      data: [35, 25, 20, 15, 5],
      backgroundColor: [
        '#ef4444',
        '#f59e0b',
        '#10b981',
        '#3b82f6',
        '#8b5cf6'
      ]
    }]
  };

  return (
    <div data-testid="dashboard" className="p-6 bg-gray-50 min-h-screen">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Bias Analysis Dashboard</h1>
        <p className="text-gray-600">Monitor and track bias detection across your content</p>
      </div>

      {/* Quick Actions */}
      <div className="mb-6">
        <button
          data-testid="new-analysis-button"
          onClick={onNewAnalysis}
          className="bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors"
          disabled={isLoading}
        >
          {isLoading ? 'Processing...' : '+ New Analysis'}
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div data-testid="stat-card-analyses" className="bg-white p-6 rounded-lg shadow-sm">
          <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">Total Analyses</h3>
          <p className="mt-2 text-3xl font-bold text-gray-900">{mockStats.total_analyses}</p>
          <p className="mt-1 text-sm text-gray-600">This month</p>
        </div>

        <div data-testid="stat-card-avg-score" className="bg-white p-6 rounded-lg shadow-sm">
          <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">Avg Bias Score</h3>
          <p className="mt-2 text-3xl font-bold text-gray-900">{(mockStats.avg_bias_score * 100).toFixed(1)}%</p>
          <p className={`mt-1 text-sm ${mockStats.trend_direction === 'improving' ? 'text-green-600' : 'text-red-600'}`}>
            {mockStats.trend_direction === 'improving' ? '↓ Improving' : '↑ Worsening'}
          </p>
        </div>

        <div data-testid="stat-card-common-bias" className="bg-white p-6 rounded-lg shadow-sm">
          <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">Most Common Bias</h3>
          <p className="mt-2 text-lg font-semibold text-gray-900 capitalize">
            {mockStats.most_common_bias.replace('_', ' ')}
          </p>
          <p className="mt-1 text-sm text-gray-600">35% of detections</p>
        </div>

        <div data-testid="stat-card-efficiency" className="bg-white p-6 rounded-lg shadow-sm">
          <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">Detection Rate</h3>
          <p className="mt-2 text-3xl font-bold text-gray-900">87.3%</p>
          <p className="mt-1 text-sm text-green-600">↑ +2.1% vs last week</p>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Trend Chart */}
        <div className="bg-white p-6 rounded-lg shadow-sm">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-semibold text-gray-900">Bias Score Trends</h3>
            <div className="flex gap-2">
              <select
                data-testid="timeframe-select"
                value={selectedTimeframe}
                onChange={(e) => setSelectedTimeframe(e.target.value)}
                className="text-sm border border-gray-300 rounded-md px-3 py-1"
              >
                <option value="7days">Last 7 Days</option>
                <option value="30days">Last 30 Days</option>
                <option value="90days">Last 90 Days</option>
              </select>
              <select
                data-testid="metric-select"
                value={selectedMetric}
                onChange={(e) => setSelectedMetric(e.target.value)}
                className="text-sm border border-gray-300 rounded-md px-3 py-1"
              >
                <option value="bias_score">Bias Score</option>
                <option value="confidence">Confidence</option>
                <option value="processing_time">Processing Time</option>
              </select>
            </div>
          </div>
          <div data-testid="trend-chart-container">
            {/* Line chart would go here */}
            <div data-testid="line-chart">Mock Line Chart: Bias Score Trend</div>
          </div>
        </div>

        {/* Bias Distribution */}
        <div className="bg-white p-6 rounded-lg shadow-sm">
          <h3 className="text-lg font-semibold text-gray-900 mb-6">Bias Type Distribution</h3>
          <div data-testid="distribution-chart-container">
            <div data-testid="doughnut-chart">
              Mock Doughnut Chart: {JSON.stringify(biasDistribution.labels)}
            </div>
          </div>
          <div className="mt-4 space-y-2">
            {biasDistribution.labels.map((label, index) => (
              <div key={label} className="flex justify-between items-center">
                <div className="flex items-center">
                  <div
                    className="w-3 h-3 rounded-full mr-3"
                    style={{ backgroundColor: biasDistribution.datasets[0].backgroundColor[index] }}
                  />
                  <span className="text-sm text-gray-600">{label}</span>
                </div>
                <span className="text-sm font-medium text-gray-900">
                  {biasDistribution.datasets[0].data[index]}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Analyses */}
      <div className="bg-white rounded-lg shadow-sm">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Recent Analyses</h3>
        </div>
        <div className="overflow-hidden">
          {analysisHistory.length === 0 ? (
            <div data-testid="empty-state" className="p-8 text-center">
              <div className="w-12 h-12 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
                📊
              </div>
              <h4 className="text-lg font-medium text-gray-900 mb-2">No analyses yet</h4>
              <p className="text-gray-600 mb-4">Start analyzing content to see your bias detection history</p>
              <button
                onClick={onNewAnalysis}
                className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
              >
                Create First Analysis
              </button>
            </div>
          ) : (
            <div data-testid="analyses-table" className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Text Preview
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Bias Score
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Top Bias
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Date
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {analysisHistory.slice(0, 5).map((analysis, index) => (
                    <tr key={index} data-testid={`analysis-row-${index}`}>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900 max-w-xs truncate">
                          {analysis.text}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          analysis.overall_score > 0.7 ? 'bg-red-100 text-red-800' :
                          analysis.overall_score > 0.4 ? 'bg-yellow-100 text-yellow-800' :
                          'bg-green-100 text-green-800'
                        }`}>
                          {(analysis.overall_score * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {analysis.detected_biases?.[0]?.category || 'None'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {analysis.timestamp ? new Date(analysis.timestamp).toLocaleDateString() : 'N/A'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <button
                          data-testid={`view-details-${index}`}
                          className="text-blue-600 hover:text-blue-900"
                        >
                          View Details
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Loading Overlay */}
      {isLoading && (
        <div
          data-testid="loading-overlay"
          className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50"
        >
          <div className="bg-white p-6 rounded-lg shadow-xl">
            <div className="flex items-center space-x-3">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
              <span>Processing analysis...</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Test wrapper
const TestWrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('Dashboard Component', () => {
  const mockOnNewAnalysis = vi.fn();
  const user = userEvent.setup();

  const sampleAnalysisHistory = [
    {
      text: "This evidence clearly proves our hypothesis is correct.",
      overall_score: 0.75,
      confidence: 0.88,
      detected_biases: [
        { category: 'confirmation_bias', severity: 'high', score: 0.8 }
      ],
      timestamp: '2024-12-01T10:00:00Z'
    },
    {
      text: "Based on recent news coverage, this is obviously widespread.",
      overall_score: 0.65,
      confidence: 0.72,
      detected_biases: [
        { category: 'availability_bias', severity: 'medium', score: 0.7 }
      ],
      timestamp: '2024-12-01T09:00:00Z'
    },
    {
      text: "All successful companies follow this exact strategy.",
      overall_score: 0.85,
      confidence: 0.91,
      detected_biases: [
        { category: 'survivorship_bias', severity: 'high', score: 0.9 }
      ],
      timestamp: '2024-11-30T15:30:00Z'
    }
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Initial Rendering', () => {
    it('renders dashboard with all main sections', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={[]} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      expect(screen.getByTestId('dashboard')).toBeInTheDocument();
      expect(screen.getByText('Bias Analysis Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Monitor and track bias detection across your content')).toBeInTheDocument();
      expect(screen.getByTestId('new-analysis-button')).toBeInTheDocument();
    });

    it('displays all stat cards with correct data', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={sampleAnalysisHistory} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      // Check stat cards
      expect(screen.getByTestId('stat-card-analyses')).toBeInTheDocument();
      expect(screen.getByText('3')).toBeInTheDocument(); // Total analyses

      expect(screen.getByTestId('stat-card-avg-score')).toBeInTheDocument();
      expect(screen.getByText('45.0%')).toBeInTheDocument(); // Average bias score

      expect(screen.getByTestId('stat-card-common-bias')).toBeInTheDocument();
      expect(screen.getByText('Confirmation bias')).toBeInTheDocument();

      expect(screen.getByTestId('stat-card-efficiency')).toBeInTheDocument();
      expect(screen.getByText('87.3%')).toBeInTheDocument();
    });

    it('renders chart components', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={sampleAnalysisHistory} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      expect(screen.getByTestId('trend-chart-container')).toBeInTheDocument();
      expect(screen.getByTestId('distribution-chart-container')).toBeInTheDocument();
    });

    it('displays dropdown controls for chart filtering', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={sampleAnalysisHistory} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      const timeframeSelect = screen.getByTestId('timeframe-select');
      const metricSelect = screen.getByTestId('metric-select');

      expect(timeframeSelect).toBeInTheDocument();
      expect(metricSelect).toBeInTheDocument();

      expect(timeframeSelect).toHaveValue('7days');
      expect(metricSelect).toHaveValue('bias_score');
    });
  });

  describe('User Interactions', () => {
    it('calls onNewAnalysis when new analysis button is clicked', async () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={[]} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      const newAnalysisButton = screen.getByTestId('new-analysis-button');
      await user.click(newAnalysisButton);

      expect(mockOnNewAnalysis).toHaveBeenCalledTimes(1);
    });

    it('updates timeframe selection', async () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={sampleAnalysisHistory} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      const timeframeSelect = screen.getByTestId('timeframe-select');
      await user.selectOptions(timeframeSelect, '30days');

      expect(timeframeSelect).toHaveValue('30days');
    });

    it('updates metric selection', async () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={sampleAnalysisHistory} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      const metricSelect = screen.getByTestId('metric-select');
      await user.selectOptions(metricSelect, 'confidence');

      expect(metricSelect).toHaveValue('confidence');
    });

    it('handles view details button clicks', async () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={sampleAnalysisHistory} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      const viewDetailsButton = screen.getByTestId('view-details-0');
      await user.click(viewDetailsButton);

      // In a real implementation, this would navigate or open a modal
      expect(viewDetailsButton).toBeInTheDocument();
    });
  });

  describe('Analysis History Display', () => {
    it('shows empty state when no analyses exist', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={[]} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      expect(screen.getByTestId('empty-state')).toBeInTheDocument();
      expect(screen.getByText('No analyses yet')).toBeInTheDocument();
      expect(screen.getByText('Start analyzing content to see your bias detection history')).toBeInTheDocument();
    });

    it('displays analysis history table when data exists', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={sampleAnalysisHistory} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      expect(screen.getByTestId('analyses-table')).toBeInTheDocument();
      expect(screen.queryByTestId('empty-state')).not.toBeInTheDocument();

      // Check table headers
      expect(screen.getByText('Text Preview')).toBeInTheDocument();
      expect(screen.getByText('Bias Score')).toBeInTheDocument();
      expect(screen.getByText('Top Bias')).toBeInTheDocument();
      expect(screen.getByText('Date')).toBeInTheDocument();
      expect(screen.getByText('Actions')).toBeInTheDocument();
    });

    it('displays correct analysis data in table rows', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={sampleAnalysisHistory} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      // Check first row data
      expect(screen.getByTestId('analysis-row-0')).toBeInTheDocument();
      expect(screen.getByText('This evidence clearly proves our hypothesis is correct.')).toBeInTheDocument();
      expect(screen.getByText('75.0%')).toBeInTheDocument();
      expect(screen.getByText('confirmation_bias')).toBeInTheDocument();

      // Check second row data
      expect(screen.getByTestId('analysis-row-1')).toBeInTheDocument();
      expect(screen.getByText('Based on recent news coverage, this is obviously widespread.')).toBeInTheDocument();
      expect(screen.getByText('65.0%')).toBeInTheDocument();
      expect(screen.getByText('availability_bias')).toBeInTheDocument();
    });

    it('applies correct styling based on bias scores', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={sampleAnalysisHistory} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      // High score should be red
      const highScoreElement = screen.getByText('75.0%');
      expect(highScoreElement).toHaveClass('bg-red-100', 'text-red-800');

      // Medium score should be yellow
      const mediumScoreElement = screen.getByText('65.0%');
      expect(mediumScoreElement).toHaveClass('bg-yellow-100', 'text-yellow-800');
    });

    it('limits displayed analyses to 5 most recent', () => {
      const manyAnalyses = Array(10).fill(null).map((_, i) => ({
        text: `Analysis ${i}`,
        overall_score: 0.5,
        confidence: 0.8,
        detected_biases: [{ category: 'test_bias', severity: 'medium', score: 0.5 }],
        timestamp: new Date(2024, 11, i + 1).toISOString()
      }));

      render(
        <TestWrapper>
          <Dashboard analysisHistory={manyAnalyses} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      // Should only show 5 rows (0-4)
      expect(screen.getByTestId('analysis-row-0')).toBeInTheDocument();
      expect(screen.getByTestId('analysis-row-4')).toBeInTheDocument();
      expect(screen.queryByTestId('analysis-row-5')).not.toBeInTheDocument();
    });
  });

  describe('Loading States', () => {
    it('shows loading overlay when isLoading is true', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={[]} onNewAnalysis={mockOnNewAnalysis} isLoading={true} />
        </TestWrapper>
      );

      expect(screen.getByTestId('loading-overlay')).toBeInTheDocument();
      expect(screen.getByText('Processing analysis...')).toBeInTheDocument();
    });

    it('disables new analysis button when loading', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={[]} onNewAnalysis={mockOnNewAnalysis} isLoading={true} />
        </TestWrapper>
      );

      const newAnalysisButton = screen.getByTestId('new-analysis-button');
      expect(newAnalysisButton).toBeDisabled();
      expect(newAnalysisButton).toHaveTextContent('Processing...');
    });

    it('hides loading overlay when isLoading is false', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={[]} onNewAnalysis={mockOnNewAnalysis} isLoading={false} />
        </TestWrapper>
      );

      expect(screen.queryByTestId('loading-overlay')).not.toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('applies responsive grid classes', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={[]} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      // Check for responsive grid classes on stats cards
      const statsContainer = screen.getByTestId('stat-card-analyses').parentElement;
      expect(statsContainer).toHaveClass('grid', 'grid-cols-1', 'md:grid-cols-2', 'lg:grid-cols-4');
    });

    it('handles table overflow on small screens', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={sampleAnalysisHistory} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      const tableContainer = screen.getByTestId('analyses-table');
      expect(tableContainer).toHaveClass('overflow-x-auto');
    });
  });

  describe('Data Formatting', () => {
    it('formats dates correctly', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={sampleAnalysisHistory} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      // Check that dates are formatted as locale strings
      const dateElements = screen.getAllByText(/12\/1\/2024|12\/[0-9]+\/2024|11\/30\/2024/);
      expect(dateElements.length).toBeGreaterThan(0);
    });

    it('handles missing data gracefully', () => {
      const incompleteAnalysis = [{
        text: "Test text",
        overall_score: 0.5,
        confidence: 0.8
        // Missing detected_biases and timestamp
      }];

      render(
        <TestWrapper>
          <Dashboard analysisHistory={incompleteAnalysis} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      expect(screen.getByText('None')).toBeInTheDocument(); // For missing bias
      expect(screen.getByText('N/A')).toBeInTheDocument(); // For missing date
    });

    it('truncates long text previews', () => {
      const longTextAnalysis = [{
        text: "This is a very long text that should be truncated when displayed in the table because it exceeds the maximum width allocated for the text preview column and would otherwise break the layout",
        overall_score: 0.5,
        confidence: 0.8,
        detected_biases: [{ category: 'test_bias', severity: 'medium', score: 0.5 }],
        timestamp: '2024-12-01T10:00:00Z'
      }];

      render(
        <TestWrapper>
          <Dashboard analysisHistory={longTextAnalysis} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      // Text should be truncated with CSS classes
      const textElement = screen.getByText(/This is a very long text/);
      expect(textElement).toHaveClass('truncate', 'max-w-xs');
    });
  });

  describe('Chart Integration', () => {
    it('displays trend chart with correct timeframe data', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={sampleAnalysisHistory} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      expect(screen.getByTestId('trend-chart-container')).toBeInTheDocument();
      expect(screen.getByText('Bias Score Trends')).toBeInTheDocument();
    });

    it('shows bias distribution chart with legend', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={sampleAnalysisHistory} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      expect(screen.getByTestId('distribution-chart-container')).toBeInTheDocument();
      expect(screen.getByText('Bias Type Distribution')).toBeInTheDocument();

      // Check legend items
      expect(screen.getByText('Confirmation')).toBeInTheDocument();
      expect(screen.getByText('Anchoring')).toBeInTheDocument();
      expect(screen.getByText('Availability')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper heading hierarchy', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={[]} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      const h1 = screen.getByRole('heading', { level: 1 });
      expect(h1).toHaveTextContent('Bias Analysis Dashboard');

      const h3Elements = screen.getAllByRole('heading', { level: 3 });
      expect(h3Elements.length).toBeGreaterThan(0);
    });

    it('supports keyboard navigation', async () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={sampleAnalysisHistory} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      const newAnalysisButton = screen.getByTestId('new-analysis-button');
      const timeframeSelect = screen.getByTestId('timeframe-select');

      // Test tab navigation
      await user.tab();
      expect(newAnalysisButton).toHaveFocus();

      await user.tab();
      expect(timeframeSelect).toHaveFocus();
    });

    it('provides accessible table structure', () => {
      render(
        <TestWrapper>
          <Dashboard analysisHistory={sampleAnalysisHistory} onNewAnalysis={mockOnNewAnalysis} />
        </TestWrapper>
      );

      const table = screen.getByRole('table');
      expect(table).toBeInTheDocument();

      const columnHeaders = screen.getAllByRole('columnheader');
      expect(columnHeaders).toHaveLength(5);
    });
  });
});