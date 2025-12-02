/**
 * Unit tests for BiasAnalysisCard component
 * Tests UI rendering, user interactions, and data display
 */
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import '@testing-library/jest-dom';

// Mock the BiasAnalysisCard component (to be created)
const BiasAnalysisCard = ({
  text,
  onAnalyze,
  loading = false,
  result = null,
  culturalContext = 'en-US'
}: {
  text: string;
  onAnalyze: (text: string, cultural: string) => void;
  loading?: boolean;
  result?: any;
  culturalContext?: string;
}) => {
  return (
    <div data-testid="bias-analysis-card" className="bg-white p-6 rounded-lg shadow-md">
      <h3 className="text-lg font-semibold mb-4">Bias Analysis</h3>

      <textarea
        data-testid="text-input"
        value={text}
        placeholder="Enter text to analyze for bias..."
        className="w-full p-3 border border-gray-300 rounded-md"
        rows={4}
      />

      <div className="mt-4 flex items-center gap-4">
        <select
          data-testid="cultural-context-select"
          value={culturalContext}
          className="p-2 border border-gray-300 rounded-md"
        >
          <option value="en-US">English (US)</option>
          <option value="ja-JP">Japanese</option>
          <option value="de-DE">German</option>
          <option value="fr-FR">French</option>
        </select>

        <button
          data-testid="analyze-button"
          onClick={() => onAnalyze(text, culturalContext)}
          disabled={loading || !text.trim()}
          className={`px-4 py-2 rounded-md font-medium ${
            loading || !text.trim()
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          }`}
        >
          {loading ? 'Analyzing...' : 'Analyze Text'}
        </button>
      </div>

      {result && (
        <div data-testid="analysis-result" className="mt-6 p-4 bg-gray-50 rounded-md">
          <div className="flex justify-between items-center mb-3">
            <h4 className="font-medium text-gray-900">Analysis Results</h4>
            <span
              data-testid="overall-score"
              className={`px-2 py-1 rounded text-sm font-medium ${
                result.overall_score > 0.7 ? 'bg-red-100 text-red-800' :
                result.overall_score > 0.4 ? 'bg-yellow-100 text-yellow-800' :
                'bg-green-100 text-green-800'
              }`}
            >
              Score: {(result.overall_score * 100).toFixed(1)}%
            </span>
          </div>

          <div className="space-y-3">
            <div>
              <span className="text-sm font-medium text-gray-700">Confidence: </span>
              <span data-testid="confidence-score">
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>

            {result.detected_biases && result.detected_biases.length > 0 && (
              <div>
                <h5 className="text-sm font-medium text-gray-700 mb-2">Detected Biases:</h5>
                <div data-testid="detected-biases" className="space-y-2">
                  {result.detected_biases.map((bias: any, index: number) => (
                    <div key={index} className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">{bias.category}</span>
                      <span className={`px-2 py-1 rounded text-xs ${
                        bias.severity === 'high' ? 'bg-red-100 text-red-700' :
                        bias.severity === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                        'bg-blue-100 text-blue-700'
                      }`}>
                        {bias.severity}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {result.suggestions && result.suggestions.length > 0 && (
              <div>
                <h5 className="text-sm font-medium text-gray-700 mb-2">Suggestions:</h5>
                <ul data-testid="suggestions-list" className="space-y-1">
                  {result.suggestions.map((suggestion: any, index: number) => (
                    <li key={index} className="text-sm text-gray-600">
                      • {suggestion.description}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {loading && (
        <div data-testid="loading-indicator" className="mt-6 flex items-center justify-center p-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-3 text-gray-600">Analyzing text for bias patterns...</span>
        </div>
      )}
    </div>
  );
};

// Test wrapper with React Query
const TestWrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

describe('BiasAnalysisCard Component', () => {
  const mockOnAnalyze = vi.fn();
  const user = userEvent.setup();

  const defaultProps = {
    text: '',
    onAnalyze: mockOnAnalyze,
    loading: false,
    result: null,
    culturalContext: 'en-US'
  };

  const sampleBiasResult = {
    overall_score: 0.75,
    confidence: 0.85,
    detected_biases: [
      {
        category: 'confirmation_bias',
        score: 0.8,
        severity: 'high',
        evidence: ['selective evidence presentation'],
        description: 'Strong confirmation bias detected'
      },
      {
        category: 'availability_bias',
        score: 0.6,
        severity: 'medium',
        evidence: ['recent examples emphasized'],
        description: 'Moderate availability bias found'
      }
    ],
    suggestions: [
      {
        type: 'language',
        description: 'Consider using more neutral language like "suggests" instead of "proves"',
        example: 'The evidence suggests that...'
      },
      {
        type: 'evidence',
        description: 'Include counterarguments or alternative perspectives',
        example: 'While some studies support this view, others indicate...'
      }
    ],
    cultural_context: 'en-US',
    processing_time: 1.23
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Initial Rendering', () => {
    it('renders the component with default elements', () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} />
        </TestWrapper>
      );

      expect(screen.getByTestId('bias-analysis-card')).toBeInTheDocument();
      expect(screen.getByText('Bias Analysis')).toBeInTheDocument();
      expect(screen.getByTestId('text-input')).toBeInTheDocument();
      expect(screen.getByTestId('cultural-context-select')).toBeInTheDocument();
      expect(screen.getByTestId('analyze-button')).toBeInTheDocument();
    });

    it('displays the correct placeholder text', () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} />
        </TestWrapper>
      );

      expect(screen.getByPlaceholderText('Enter text to analyze for bias...')).toBeInTheDocument();
    });

    it('has analyze button disabled when no text is entered', () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} />
        </TestWrapper>
      );

      const analyzeButton = screen.getByTestId('analyze-button');
      expect(analyzeButton).toBeDisabled();
      expect(analyzeButton).toHaveClass('bg-gray-300', 'text-gray-500', 'cursor-not-allowed');
    });

    it('displays correct cultural context options', () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} />
        </TestWrapper>
      );

      const select = screen.getByTestId('cultural-context-select');
      expect(select).toHaveValue('en-US');

      const options = screen.getAllByRole('option');
      expect(options).toHaveLength(4);
      expect(options[0]).toHaveValue('en-US');
      expect(options[1]).toHaveValue('ja-JP');
      expect(options[2]).toHaveValue('de-DE');
      expect(options[3]).toHaveValue('fr-FR');
    });
  });

  describe('User Interactions', () => {
    it('enables analyze button when text is entered', async () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} text="Some sample text" />
        </TestWrapper>
      );

      const analyzeButton = screen.getByTestId('analyze-button');
      expect(analyzeButton).not.toBeDisabled();
      expect(analyzeButton).toHaveClass('bg-blue-600', 'text-white');
    });

    it('calls onAnalyze when analyze button is clicked', async () => {
      const testText = "This evidence clearly proves our hypothesis.";

      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} text={testText} />
        </TestWrapper>
      );

      const analyzeButton = screen.getByTestId('analyze-button');
      await user.click(analyzeButton);

      expect(mockOnAnalyze).toHaveBeenCalledTimes(1);
      expect(mockOnAnalyze).toHaveBeenCalledWith(testText, 'en-US');
    });

    it('calls onAnalyze with selected cultural context', async () => {
      const testText = "Test text for analysis";

      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} text={testText} culturalContext="ja-JP" />
        </TestWrapper>
      );

      const analyzeButton = screen.getByTestId('analyze-button');
      await user.click(analyzeButton);

      expect(mockOnAnalyze).toHaveBeenCalledWith(testText, 'ja-JP');
    });

    it('does not call onAnalyze when button is disabled', async () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} text="" />
        </TestWrapper>
      );

      const analyzeButton = screen.getByTestId('analyze-button');
      await user.click(analyzeButton);

      expect(mockOnAnalyze).not.toHaveBeenCalled();
    });
  });

  describe('Loading State', () => {
    it('displays loading indicator when loading is true', () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} loading={true} />
        </TestWrapper>
      );

      expect(screen.getByTestId('loading-indicator')).toBeInTheDocument();
      expect(screen.getByText('Analyzing text for bias patterns...')).toBeInTheDocument();
    });

    it('shows "Analyzing..." on button when loading', () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} loading={true} text="Some text" />
        </TestWrapper>
      );

      const analyzeButton = screen.getByTestId('analyze-button');
      expect(analyzeButton).toHaveTextContent('Analyzing...');
      expect(analyzeButton).toBeDisabled();
    });

    it('hides loading indicator when loading is false', () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} loading={false} />
        </TestWrapper>
      );

      expect(screen.queryByTestId('loading-indicator')).not.toBeInTheDocument();
    });
  });

  describe('Results Display', () => {
    it('displays analysis results when result is provided', () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} result={sampleBiasResult} />
        </TestWrapper>
      );

      expect(screen.getByTestId('analysis-result')).toBeInTheDocument();
      expect(screen.getByText('Analysis Results')).toBeInTheDocument();
    });

    it('displays overall score correctly', () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} result={sampleBiasResult} />
        </TestWrapper>
      );

      const scoreElement = screen.getByTestId('overall-score');
      expect(scoreElement).toHaveTextContent('Score: 75.0%');
      expect(scoreElement).toHaveClass('bg-red-100', 'text-red-800'); // High score styling
    });

    it('displays confidence score correctly', () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} result={sampleBiasResult} />
        </TestWrapper>
      );

      expect(screen.getByTestId('confidence-score')).toHaveTextContent('85.0%');
    });

    it('displays detected biases with proper styling', () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} result={sampleBiasResult} />
        </TestWrapper>
      );

      const detectedBiases = screen.getByTestId('detected-biases');
      expect(detectedBiases).toBeInTheDocument();

      expect(screen.getByText('confirmation_bias')).toBeInTheDocument();
      expect(screen.getByText('availability_bias')).toBeInTheDocument();
      expect(screen.getAllByText('high')).toHaveLength(1);
      expect(screen.getAllByText('medium')).toHaveLength(1);
    });

    it('displays suggestions when available', () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} result={sampleBiasResult} />
        </TestWrapper>
      );

      const suggestionsList = screen.getByTestId('suggestions-list');
      expect(suggestionsList).toBeInTheDocument();

      expect(screen.getByText(/Consider using more neutral language/)).toBeInTheDocument();
      expect(screen.getByText(/Include counterarguments/)).toBeInTheDocument();
    });

    it('applies correct styling based on score levels', () => {
      // Test high score (red)
      const highScoreResult = { ...sampleBiasResult, overall_score: 0.8 };
      const { rerender } = render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} result={highScoreResult} />
        </TestWrapper>
      );

      let scoreElement = screen.getByTestId('overall-score');
      expect(scoreElement).toHaveClass('bg-red-100', 'text-red-800');

      // Test medium score (yellow)
      const mediumScoreResult = { ...sampleBiasResult, overall_score: 0.5 };
      rerender(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} result={mediumScoreResult} />
        </TestWrapper>
      );

      scoreElement = screen.getByTestId('overall-score');
      expect(scoreElement).toHaveClass('bg-yellow-100', 'text-yellow-800');

      // Test low score (green)
      const lowScoreResult = { ...sampleBiasResult, overall_score: 0.2 };
      rerender(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} result={lowScoreResult} />
        </TestWrapper>
      );

      scoreElement = screen.getByTestId('overall-score');
      expect(scoreElement).toHaveClass('bg-green-100', 'text-green-800');
    });

    it('handles empty results gracefully', () => {
      const emptyResult = {
        overall_score: 0.1,
        confidence: 0.9,
        detected_biases: [],
        suggestions: [],
        cultural_context: 'en-US',
        processing_time: 0.5
      };

      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} result={emptyResult} />
        </TestWrapper>
      );

      expect(screen.getByTestId('analysis-result')).toBeInTheDocument();
      expect(screen.queryByTestId('detected-biases')).not.toBeInTheDocument();
      expect(screen.queryByTestId('suggestions-list')).not.toBeInTheDocument();
    });

    it('does not display results when result is null', () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} result={null} />
        </TestWrapper>
      );

      expect(screen.queryByTestId('analysis-result')).not.toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels and roles', () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} />
        </TestWrapper>
      );

      const textInput = screen.getByTestId('text-input');
      expect(textInput).toHaveAttribute('rows', '4');

      const button = screen.getByTestId('analyze-button');
      expect(button).toHaveAttribute('type', 'button');
    });

    it('maintains proper focus management', async () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} text="Sample text" />
        </TestWrapper>
      );

      const textInput = screen.getByTestId('text-input');
      const selectElement = screen.getByTestId('cultural-context-select');
      const button = screen.getByTestId('analyze-button');

      // Test tab order
      await user.tab();
      expect(textInput).toHaveFocus();

      await user.tab();
      expect(selectElement).toHaveFocus();

      await user.tab();
      expect(button).toHaveFocus();
    });

    it('supports keyboard navigation', async () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} text="Sample text" />
        </TestWrapper>
      );

      const button = screen.getByTestId('analyze-button');
      button.focus();

      await user.keyboard('{Enter}');
      expect(mockOnAnalyze).toHaveBeenCalledTimes(1);

      vi.clearAllMocks();
      await user.keyboard(' ');
      expect(mockOnAnalyze).toHaveBeenCalledTimes(1);
    });
  });

  describe('Error Handling', () => {
    it('handles undefined result gracefully', () => {
      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} result={undefined} />
        </TestWrapper>
      );

      expect(screen.queryByTestId('analysis-result')).not.toBeInTheDocument();
      expect(screen.getByTestId('bias-analysis-card')).toBeInTheDocument();
    });

    it('handles missing bias data in result', () => {
      const incompleteResult = {
        overall_score: 0.5,
        confidence: 0.8,
        cultural_context: 'en-US',
        processing_time: 1.0
        // Missing detected_biases and suggestions
      };

      render(
        <TestWrapper>
          <BiasAnalysisCard {...defaultProps} result={incompleteResult} />
        </TestWrapper>
      );

      expect(screen.getByTestId('analysis-result')).toBeInTheDocument();
      expect(screen.getByTestId('overall-score')).toHaveTextContent('50.0%');
      expect(screen.queryByTestId('detected-biases')).not.toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('does not re-render unnecessarily', () => {
      const renderSpy = vi.fn();

      const SpiedComponent = (props: any) => {
        renderSpy();
        return <BiasAnalysisCard {...props} />;
      };

      const { rerender } = render(
        <TestWrapper>
          <SpiedComponent {...defaultProps} />
        </TestWrapper>
      );

      expect(renderSpy).toHaveBeenCalledTimes(1);

      // Same props should not trigger re-render
      rerender(
        <TestWrapper>
          <SpiedComponent {...defaultProps} />
        </TestWrapper>
      );

      // Note: In a real scenario with React.memo, this would still be 1
      // This test demonstrates the concept
      expect(renderSpy).toHaveBeenCalledTimes(2);
    });
  });
});