import React from 'react';
import { createRoot } from 'react-dom/client';

import { SplunkThemeProvider } from '@splunk/themes';
import { getUserTheme, getThemeOptions } from '@splunk/splunk-utils/themes';

import UrcApp from '../src/UrcApp';

getUserTheme()
    .then((theme) => {
        const containerEl = document.getElementById('main-component-container');
        const splunkTheme = getThemeOptions(theme);
        const root = createRoot(containerEl);

        root.render(
            <SplunkThemeProvider {...splunkTheme}>
                <UrcApp />
            </SplunkThemeProvider>
        );
    })
    .catch((e) => {
        const errorEl = document.createElement('span');
        errorEl.innerHTML = e;
        document.body.appendChild(errorEl);
    });
