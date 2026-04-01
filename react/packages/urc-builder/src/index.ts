// Public exports for @splunk/urc-builder

// Screen components
export { default as StreamDashboard } from './dashboard/StreamDashboard';
export { default as StreamBuilder } from './builder/StreamBuilder';

// Context providers
export { BuilderProvider, useBuilder } from './context/BuilderContext';
export { DashboardProvider, useDashboard } from './context/DashboardContext';

// Types
export type {
    InputSummary,
    SavedInput,
    AccountSummary,
    ValidationResult,
    TestResult,
    DetectedField,
    TimestampCandidate,
    StreamConfig,
    BuilderState,
    BuilderAction,
    DashboardState,
    InputPayload,
    FormFieldDef,
    ComponentFormDef,
} from './types';
