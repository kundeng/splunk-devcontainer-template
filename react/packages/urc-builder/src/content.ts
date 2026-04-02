/**
 * content.ts — Centralized UI copy, icons, and help text.
 *
 * Single source of truth for all user-facing strings in the URC builder.
 * Components import from here instead of hardcoding labels and descriptions.
 */

import React from 'react';
import Globe from '@splunk/react-icons/Globe';
import Cogs from '@splunk/react-icons/Cogs';
import Cylinder from '@splunk/react-icons/Cylinder';
import CheckCircle from '@splunk/react-icons/CheckCircle';
import CrossCircle from '@splunk/react-icons/CrossCircle';
import IdentityCard from '@splunk/react-icons/IdentityCard';
import Filter from '@splunk/react-icons/Filter';
import LinesThree from '@splunk/react-icons/LinesThree';
import Clock from '@splunk/react-icons/Clock';
import Wrench from '@splunk/react-icons/Wrench';
import Shield from '@splunk/react-icons/Shield';
import Calendar from '@splunk/react-icons/Calendar';
import Key from '@splunk/react-icons/Key';
import Plus from '@splunk/react-icons/Plus';
import ControlPlayCircle from '@splunk/react-icons/ControlPlayCircle';
import NodeBranch from '@splunk/react-icons/NodeBranch';
import FileText from '@splunk/react-icons/FileText';

// ── Helpers ──

/** Create an icon element with consistent sizing. */
function icon(component: any, size = 20) {
    return React.createElement(component, { width: size, height: size });
}

function iconSm(component: any) {
    return icon(component, 16);
}

// ── Types ──

interface StepDef {
    label: string;
    icon: React.ReactElement;
    description: string;
}

interface SectionDef {
    title: string;
    icon: React.ReactElement;
    description: string;
}

interface FieldDef {
    label: string;
    help: string;
    placeholder?: string;
}

// ── Wizard Steps ──

export const STEPS: StepDef[] = [
    {
        label: 'Connect',
        icon: icon(Globe),
        description: 'Identify your stream and configure the API connection.',
    },
    {
        label: 'Configure',
        icon: icon(Cogs),
        description: 'Define how records are extracted, paginated, and transformed.',
    },
    {
        label: 'Output',
        icon: icon(Cylinder),
        description: 'Choose where data lands in Splunk and how often to collect.',
    },
    {
        label: 'Test & Save',
        icon: icon(CheckCircle),
        description: 'Validate the connection and save your stream configuration.',
    },
];

// ── Section Headers ──

export const SECTIONS = {
    streamIdentity: {
        title: 'Stream Identity',
        icon: icon(IdentityCard),
        description: 'Give this data stream a unique name.',
    } as SectionDef,

    apiConnection: {
        title: 'API Connection',
        icon: icon(Globe),
        description: 'Configure the API endpoint and authentication.',
    } as SectionDef,

    recordSelector: {
        title: 'Record Selector',
        icon: icon(Filter),
        description: 'Define how individual records are extracted from the API response.',
    } as SectionDef,

    pagination: {
        title: 'Pagination',
        icon: icon(LinesThree),
        description: 'Configure how to retrieve multiple pages of results.',
    } as SectionDef,

    incrementalSync: {
        title: 'Incremental Sync',
        icon: icon(Clock),
        description: 'Track a cursor field to only fetch new or updated records.',
    } as SectionDef,

    transformations: {
        title: 'Transformations',
        icon: icon(Wrench),
        description: 'Apply transformations to records before indexing.',
    } as SectionDef,

    errorHandling: {
        title: 'Error Handling',
        icon: icon(Shield),
        description: 'Configure retry behavior and error response handling.',
    } as SectionDef,

    destination: {
        title: 'Destination',
        icon: icon(Cylinder),
        description: 'Choose the Splunk index and sourcetype for collected events.',
    } as SectionDef,

    schedule: {
        title: 'Schedule',
        icon: icon(Calendar),
        description: 'Set the collection interval and organize with tags.',
    } as SectionDef,

    partitionRouter: {
        title: 'Partition Router',
        icon: icon(NodeBranch),
        description: 'Split collection into partitions. Use SubstreamPartitionRouter for parent-child relationships (e.g. repos then issues per repo).',
    } as SectionDef,

    decoder: {
        title: 'Decoder',
        icon: icon(FileText),
        description: 'Choose how to decode the API response. Default is JSON. Use CSV, XML, or compressed decoders for non-JSON APIs.',
    } as SectionDef,

    testConnection: {
        title: 'Test Connection',
        icon: icon(ControlPlayCircle),
        description: 'Run a live test against the API to verify your configuration.',
    } as SectionDef,
};

// ── Form Fields ──

export const FIELDS = {
    streamName: {
        label: 'Stream Name',
        help: 'A unique identifier for this data stream (letters, numbers, underscores).',
        placeholder: 'e.g. github_issues, jira_tickets',
    } as FieldDef,

    account: {
        label: 'Account',
        help: 'Select the authentication account for this API. Create accounts in Configuration.',
        placeholder: '(none — no authentication)',
    } as FieldDef,

    baseUrl: {
        label: 'Base URL',
        help: 'The root URL of the API endpoint to collect from.',
        placeholder: 'https://api.example.com/v1/resources',
    } as FieldDef,

    httpMethod: {
        label: 'HTTP Method',
        help: 'HTTP method for API requests. Use POST only for APIs that require it.',
    } as FieldDef,

    index: {
        label: 'Index',
        help: 'Splunk index where collected events will be stored.',
    } as FieldDef,

    sourcetype: {
        label: 'Sourcetype',
        help: 'Event sourcetype for parsing and search. Default: urc:api:json.',
    } as FieldDef,

    sourceOverride: {
        label: 'Source Override',
        help: 'Optional. Override the event source field.',
        placeholder: 'Leave empty for default',
    } as FieldDef,

    hostOverride: {
        label: 'Host Override',
        help: 'Optional. Override the event host field.',
        placeholder: 'Leave empty for default',
    } as FieldDef,

    interval: {
        label: 'Interval (seconds)',
        help: 'How often to collect data. Range: 10–86,400 seconds (10s to 24h).',
    } as FieldDef,

    tags: {
        label: 'Tags',
        help: 'Comma-separated tags for organizing and filtering streams in the dashboard.',
        placeholder: 'e.g. devops, security',
    } as FieldDef,

    debug: {
        label: 'Debug Logging',
        help: 'Enable verbose logging for troubleshooting. Logs method entry/exit, URLs, and stream slice details.',
    } as FieldDef,
};

// ── Dashboard Copy ──

export const DASHBOARD = {
    title: 'REST API Streams',
    subtitle: 'Configure and monitor data collection from REST APIs',
    newStreamCta: 'New Stream',

    emptyState: {
        heading: 'No API streams configured yet',
        description: 'Create your first stream to start collecting data from a REST API. The wizard will guide you through connection, configuration, and testing.',
        cta: 'Create Stream',
    },

    cards: {
        total: { label: 'Total Streams', icon: icon(Globe) },
        active: { label: 'Active', icon: icon(CheckCircle) },
        disabled: { label: 'Disabled', icon: icon(CrossCircle) },
    },

    statusFilter: {
        all: 'All',
        active: 'Active',
        disabled: 'Disabled',
    },
};
