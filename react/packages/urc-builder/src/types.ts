// Shared TypeScript interfaces for URC Builder UI

// ── Schema types (imported from form-schema.ts when generated) ──

export interface FormFieldDef {
    key: string;
    label: string;
    helpText: string;
    type: 'text' | 'number' | 'boolean' | 'enum' | 'object' | 'array' | 'template';
    required: boolean;
    defaultValue?: any;
    enumValues?: { value: string; label: string }[];
    examples?: string[];
    nestedFields?: FormFieldDef[];
    interpolationContext?: string[];
}

export interface ComponentFormDef {
    type: string;
    category: string;
    label: string;
    description: string;
    fields: FormFieldDef[];
    supported: boolean;
}

// ── REST API response types ──

export interface InputSummary {
    name: string;
    account: string;
    baseUrl: string;
    interval: number;
    index: string;
    sourcetype: string;
    tags: string;
    disabled: boolean;
    manifest: string;
    debug: boolean;
}

export interface SavedInput extends InputSummary {
    parsedManifest: Record<string, any>;
}

export interface AccountSummary {
    name: string;
    authType: string;
}

export interface ValidationResult {
    path: string;
    type: string;
    severity: 'error' | 'warning' | 'info';
    message: string;
}

export interface TestResult {
    status: 'success' | 'error';
    message?: string;
    records: { stream: string; data: Record<string, any> }[];
    recordCount: number;
    validationResults: ValidationResult[];
    detectedFields: DetectedField[];
    timestampCandidates: TimestampCandidate[];
}

export interface DetectedField {
    name: string;
    type: string;
    sample: any;
}

export interface TimestampCandidate {
    field: string;
    sampleValue: string;
    parsedPreview: string;
}

// ── Builder form state ──

export interface StreamConfig {
    name: string;
    retriever: {
        type: string;
        requester: Record<string, any>;
        recordSelector: Record<string, any>;
        paginator?: Record<string, any>;
    };
    incrementalSync?: Record<string, any>;
    transformations?: Record<string, any>[];
    eventTimestamp?: Record<string, any>;
}

export interface BuilderState {
    mode: 'create' | 'edit';
    inputName: string | null;

    // Tab 1: Connection
    account: string;
    baseUrl: string;
    httpMethod: 'GET' | 'POST';

    // Tab 2: Data Mapping
    streams: StreamConfig[];

    // Tab 3: Splunk Output
    index: string;
    sourcetype: string;
    sourceOverride: string;
    hostOverride: string;
    timestampField: string | null;
    timestampFormat: string;

    // Tab 4: Schedule & Tags
    interval: number;
    tags: string;
    enabled: boolean;
    debug: boolean;

    // Tab 5: Test (ephemeral)
    testResult: TestResult | null;
    testLoading: boolean;

    // Validation
    validationResults: ValidationResult[];
    isDirty: boolean;
}

export type BuilderAction =
    | { type: 'SET_FIELD'; path: string; value: any }
    | { type: 'LOAD_INPUT'; input: SavedInput }
    | { type: 'SET_TEST_RESULT'; result: TestResult }
    | { type: 'SET_VALIDATION'; results: ValidationResult[] }
    | { type: 'SET_TEST_LOADING'; loading: boolean }
    | { type: 'RESET' };

// ── Dashboard state ──

export interface DashboardState {
    inputs: InputSummary[];
    selectedTags: string[];
    selectedStatus: 'all' | 'active' | 'error' | 'disabled';
    selectedRows: string[];
    loading: boolean;
    error: string | null;
}

// ── UCC input payload (for create/update) ──

export interface InputPayload {
    name: string;
    account: string;
    base_url: string;
    manifest: string;
    interval: string;
    index?: string;
    sourcetype?: string;
    tags?: string;
    debug?: string;
    disabled?: string;
}
