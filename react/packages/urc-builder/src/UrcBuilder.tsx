import React, { useState } from 'react';
import Button from '@splunk/react-ui/Button';

import { StyledContainer, StyledGreeting } from './UrcBuilderStyles';

const UrcBuilder = () => {
    const [counter, setCounter] = useState(0);

    const message =
        counter === 0
            ? 'You should try clicking the button.'
            : `You've clicked the button ${counter} time${counter > 1 ? 's' : ''}.`;

    return (
        <StyledContainer>
            <StyledGreeting data-testid="greeting">
                Hello, from inside UrcBuilder!
            </StyledGreeting>
            <div data-testid="message">{message}</div>
            <Button
                label="Click here"
                appearance="primary"
                onClick={() => {
                    setCounter(counter + 1);
                }}
            />
        </StyledContainer>
    );
};

export default UrcBuilder;
