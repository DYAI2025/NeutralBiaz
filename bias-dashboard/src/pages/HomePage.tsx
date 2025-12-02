import React from 'react';
import { Link } from 'react-router-dom';
import {
  DocumentTextIcon,
  ChartBarIcon,
  ShieldCheckIcon,
  ArrowRightIcon
} from '@heroicons/react/24/outline';
import { useDashboardStats } from '../hooks/useBiasDetection';

const HomePage: React.FC = () => {
  const { data: stats, isLoading } = useDashboardStats();

  const quickActions = [
    {
      title: 'Analyze Text',
      description: 'Upload or paste text for bias analysis',
      icon: DocumentTextIcon,
      href: '/analysis',
      color: 'bg-primary-600 hover:bg-primary-700',
    },
    {
      title: 'View History',
      description: 'Browse your previous analyses',
      icon: ChartBarIcon,
      href: '/history',
      color: 'bg-green-600 hover:bg-green-700',
    },
    {
      title: 'Learn More',
      description: 'Understanding bias detection',
      icon: ShieldCheckIcon,
      href: '/about',
      color: 'bg-purple-600 hover:bg-purple-700',
    },
  ];

  return (
    <div className="max-w-4xl mx-auto">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Welcome to BiasNeutralize AI
        </h1>
        <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
          Detect and neutralize bias in your text with advanced AI technology.
          Create more inclusive and fair content automatically.
        </p>
        <Link
          to="/analysis"
          className="inline-flex items-center space-x-2 bg-primary-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-primary-700 transition-colors"
        >
          <span>Start Analyzing</span>
          <ArrowRightIcon className="h-5 w-5" />
        </Link>
      </div>

      {/* Statistics */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
            <div className="text-2xl font-bold text-primary-600 mb-1">
              {stats.total_analyses.toLocaleString()}
            </div>
            <div className="text-sm text-gray-600">Total Analyses</div>
          </div>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
            <div className="text-2xl font-bold text-green-600 mb-1">
              {Math.round(stats.average_bias_score * 100)}%
            </div>
            <div className="text-sm text-gray-600">Avg Bias Score</div>
          </div>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
            <div className="text-2xl font-bold text-orange-600 mb-1">
              {stats.most_common_bias}
            </div>
            <div className="text-sm text-gray-600">Top Bias Type</div>
          </div>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
            <div className="text-2xl font-bold text-purple-600 mb-1">
              {stats.analyses_today.toLocaleString()}
            </div>
            <div className="text-sm text-gray-600">Today's Analyses</div>
          </div>
        </div>
      )}

      {isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="animate-pulse">
                <div className="h-8 bg-gray-300 rounded mb-2"></div>
                <div className="h-4 bg-gray-300 rounded"></div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Quick Actions */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
          Quick Actions
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {quickActions.map((action) => {
            const Icon = action.icon;
            return (
              <Link
                key={action.title}
                to={action.href}
                className="group block"
              >
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
                  <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg mb-4 ${action.color}`}>
                    <Icon className="h-6 w-6 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {action.title}
                  </h3>
                  <p className="text-gray-600 text-sm mb-3">
                    {action.description}
                  </p>
                  <div className="flex items-center text-primary-600 text-sm font-medium group-hover:text-primary-700">
                    <span>Get started</span>
                    <ArrowRightIcon className="h-4 w-4 ml-1" />
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Features */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
          Key Features
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-4">
            <div className="flex items-start space-x-3">
              <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <div className="w-2 h-2 bg-green-600 rounded-full"></div>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Real-time Analysis</h3>
                <p className="text-gray-600 text-sm">
                  Get instant feedback on bias detection and neutralization suggestions
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Cultural Context</h3>
                <p className="text-gray-600 text-sm">
                  Understand cultural implications and context-aware recommendations
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-6 h-6 bg-purple-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <div className="w-2 h-2 bg-purple-600 rounded-full"></div>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Visual Heatmaps</h3>
                <p className="text-gray-600 text-sm">
                  See bias markers highlighted directly in your text with severity indicators
                </p>
              </div>
            </div>
          </div>
          <div className="space-y-4">
            <div className="flex items-start space-x-3">
              <div className="w-6 h-6 bg-orange-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <div className="w-2 h-2 bg-orange-600 rounded-full"></div>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Smart Suggestions</h3>
                <p className="text-gray-600 text-sm">
                  Receive specific recommendations for creating more inclusive content
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-6 h-6 bg-red-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <div className="w-2 h-2 bg-red-600 rounded-full"></div>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Analytics Dashboard</h3>
                <p className="text-gray-600 text-sm">
                  Track your progress and analyze patterns in your writing over time
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-6 h-6 bg-yellow-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <div className="w-2 h-2 bg-yellow-600 rounded-full"></div>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Export Options</h3>
                <p className="text-gray-600 text-sm">
                  Export your neutralized content and analysis reports in various formats
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;