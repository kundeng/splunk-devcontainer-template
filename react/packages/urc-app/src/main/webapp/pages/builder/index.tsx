import React from 'react';
import layout from '@splunk/react-page';
import { getUserTheme } from '@splunk/splunk-utils/themes';
import UrcBuilder from '@splunk/urc-builder';

import { StyledContainer } from './Styles';

getUserTheme()
    .then((theme) => {
        layout(
            <StyledContainer>
                <UrcBuilder />
            </StyledContainer>,
            {
                theme,
            }
        );
    })
    .catch((e) => {
        const errorEl = document.createElement('span');
        errorEl.innerHTML = e;
        document.body.appendChild(errorEl);
    });
