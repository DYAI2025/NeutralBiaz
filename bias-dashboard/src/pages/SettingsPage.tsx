import React, { useState } from 'react';
import {
  CogIcon,
  GlobeAltIcon,
  ShieldCheckIcon,
  BellIcon,
  UserCircleIcon,
  KeyIcon
} from '@heroicons/react/24/outline';

const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState({
    // Analysis settings
    includeCulturalContext: true,
    includeSeverityAnalysis: true,
    defaultLanguage: 'en',
    autoNeutralize: false,

    // UI preferences
    theme: 'light',
    compactView: false,
    showConfidenceScores: true,
    highlightSeverity: true,

    // Notifications
    emailNotifications: true,
    analysisComplete: true,
    weeklyDigest: false,

    // Privacy
    saveAnalysisHistory: true,
    shareAnonymousData: false,
    exportDataFormat: 'json'
  });

  const handleSettingChange = (setting: string, value: boolean | string) => {
    setSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  const handleSaveSettings = () => {
    // In a real app, this would save to backend
    console.log('Saving settings:', settings);
    alert('Settings saved successfully!');
  };

  const handleResetSettings = () => {
    if (confirm('Are you sure you want to reset all settings to default?')) {
      setSettings({
        includeCulturalContext: true,
        includeSeverityAnalysis: true,
        defaultLanguage: 'en',
        autoNeutralize: false,
        theme: 'light',
        compactView: false,
        showConfidenceScores: true,
        highlightSeverity: true,
        emailNotifications: true,
        analysisComplete: true,
        weeklyDigest: false,
        saveAnalysisHistory: true,
        shareAnonymousData: false,
        exportDataFormat: 'json'
      });
    }
  };

  const SettingSection: React.FC<{
    title: string;
    description: string;
    icon: React.ElementType;
    children: React.ReactNode;
  }> = ({ title, description, icon: Icon, children }) => (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center space-x-3 mb-4">
        <Icon className="h-5 w-5 text-primary-600" />
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
      </div>
      <p className="text-gray-600 text-sm mb-6">{description}</p>
      <div className="space-y-4">{children}</div>
    </div>
  );

  const ToggleSetting: React.FC<{
    label: string;
    description?: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
  }> = ({ label, description, checked, onChange }) => (
    <div className="flex items-center justify-between">
      <div className="flex-1">
        <div className="font-medium text-gray-900">{label}</div>
        {description && (
          <div className="text-sm text-gray-600">{description}</div>
        )}
      </div>
      <label className="relative inline-flex items-center cursor-pointer">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="sr-only peer"
        />
        <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
      </label>
    </div>
  );

  const SelectSetting: React.FC<{
    label: string;
    description?: string;
    value: string;
    options: { value: string; label: string }[];
    onChange: (value: string) => void;
  }> = ({ label, description, value, options, onChange }) => (
    <div className="flex items-center justify-between">
      <div className="flex-1 mr-4">
        <div className="font-medium text-gray-900">{label}</div>
        {description && (
          <div className="text-sm text-gray-600">{description}</div>
        )}
      </div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="block w-48 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
      >
        {options.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Settings</h1>
        <p className="text-gray-600">
          Customize your bias analysis experience and preferences
        </p>
      </div>

      <div className="space-y-6">
        {/* Analysis Settings */}
        <SettingSection
          title="Analysis Settings"
          description="Configure how bias detection and analysis works"
          icon={ShieldCheckIcon}
        >
          <ToggleSetting
            label="Include Cultural Context"
            description="Analyze cultural bias and provide cultural sensitivity recommendations"
            checked={settings.includeCulturalContext}
            onChange={(checked) => handleSettingChange('includeCulturalContext', checked)}
          />
          <ToggleSetting
            label="Include Severity Analysis"
            description="Provide detailed severity ratings for detected bias markers"
            checked={settings.includeSeverityAnalysis}
            onChange={(checked) => handleSettingChange('includeSeverityAnalysis', checked)}
          />
          <SelectSetting
            label="Default Language"
            description="Primary language for text analysis"
            value={settings.defaultLanguage}
            options={[
              { value: 'en', label: 'English' },
              { value: 'es', label: 'Spanish' },
              { value: 'fr', label: 'French' },
              { value: 'de', label: 'German' }
            ]}
            onChange={(value) => handleSettingChange('defaultLanguage', value)}
          />
          <ToggleSetting
            label="Auto-neutralize Text"
            description="Automatically generate neutralized text for every analysis"
            checked={settings.autoNeutralize}
            onChange={(checked) => handleSettingChange('autoNeutralize', checked)}
          />
        </SettingSection>

        {/* Display Preferences */}
        <SettingSection
          title="Display Preferences"
          description="Customize how analysis results are displayed"
          icon={CogIcon}
        >
          <SelectSetting
            label="Theme"
            description="Choose your preferred color scheme"
            value={settings.theme}
            options={[
              { value: 'light', label: 'Light' },
              { value: 'dark', label: 'Dark' },
              { value: 'auto', label: 'Auto (System)' }
            ]}
            onChange={(value) => handleSettingChange('theme', value)}
          />
          <ToggleSetting
            label="Compact View"
            description="Use more condensed layouts for analysis results"
            checked={settings.compactView}
            onChange={(checked) => handleSettingChange('compactView', checked)}
          />
          <ToggleSetting
            label="Show Confidence Scores"
            description="Display confidence percentages for bias markers"
            checked={settings.showConfidenceScores}
            onChange={(checked) => handleSettingChange('showConfidenceScores', checked)}
          />
          <ToggleSetting
            label="Highlight by Severity"
            description="Use different colors to indicate bias severity levels"
            checked={settings.highlightSeverity}
            onChange={(checked) => handleSettingChange('highlightSeverity', checked)}
          />
        </SettingSection>

        {/* Notifications */}
        <SettingSection
          title="Notifications"
          description="Manage how and when you receive notifications"
          icon={BellIcon}
        >
          <ToggleSetting
            label="Email Notifications"
            description="Receive notifications via email"
            checked={settings.emailNotifications}
            onChange={(checked) => handleSettingChange('emailNotifications', checked)}
          />
          <ToggleSetting
            label="Analysis Complete"
            description="Notify when long-running analyses are finished"
            checked={settings.analysisComplete}
            onChange={(checked) => handleSettingChange('analysisComplete', checked)}
          />
          <ToggleSetting
            label="Weekly Digest"
            description="Receive weekly summary of your analysis activity"
            checked={settings.weeklyDigest}
            onChange={(checked) => handleSettingChange('weeklyDigest', checked)}
          />
        </SettingSection>

        {/* Privacy & Data */}
        <SettingSection
          title="Privacy & Data"
          description="Control your data privacy and export preferences"
          icon={KeyIcon}
        >
          <ToggleSetting
            label="Save Analysis History"
            description="Store your analysis results for future reference"
            checked={settings.saveAnalysisHistory}
            onChange={(checked) => handleSettingChange('saveAnalysisHistory', checked)}
          />
          <ToggleSetting
            label="Share Anonymous Data"
            description="Help improve the service by sharing anonymized usage data"
            checked={settings.shareAnonymousData}
            onChange={(checked) => handleSettingChange('shareAnonymousData', checked)}
          />
          <SelectSetting
            label="Export Format"
            description="Default format for exporting analysis data"
            value={settings.exportDataFormat}
            options={[
              { value: 'json', label: 'JSON' },
              { value: 'csv', label: 'CSV' },
              { value: 'pdf', label: 'PDF Report' }
            ]}
            onChange={(value) => handleSettingChange('exportDataFormat', value)}
          />
        </SettingSection>

        {/* Action Buttons */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-3">
            <button
              onClick={handleSaveSettings}
              className="flex-1 sm:flex-none bg-primary-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-primary-700 transition-colors"
            >
              Save Settings
            </button>
            <button
              onClick={handleResetSettings}
              className="flex-1 sm:flex-none bg-gray-100 text-gray-700 px-6 py-3 rounded-lg font-medium hover:bg-gray-200 transition-colors"
            >
              Reset to Defaults
            </button>
          </div>
          <p className="text-sm text-gray-600 mt-3">
            Settings are automatically synced across your devices when signed in.
          </p>
        </div>
      </div>
    </div>
  );
};

export default SettingsPage;