/**
 * StreamBuilder — Wizard-style stream builder with StepBar navigation.
 *
 * 4 steps: Connect → Configure → Output → Test & Save
 */

import React, { useState, useEffect, useCallback } from 'react';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import StepBar from '@splunk/react-ui/StepBar';
import Button from '@splunk/react-ui/Button';
import Breadcrumbs from '@splunk/react-ui/Breadcrumbs';
import WaitSpinner from '@splunk/react-ui/WaitSpinner';
import Message from '@splunk/react-ui/Message';
import Divider from '@splunk/react-ui/Divider';
import ChevronLeft from '@splunk/react-icons/ChevronLeft';

import { BuilderProvider, useBuilder } from '../context/BuilderContext';
import { getInput, createInput, updateInput, toggleInput } from '../services/splunk-api';
import { formStateToManifest } from '../services/manifest-serializer';
import type { InputPayload } from '../types';
import { STEPS } from '../content';
import { ConnectionTab } from './tabs/ConnectionTab';
import { DataMappingTab } from './tabs/DataMappingTab';
import SplunkOutputTab from './tabs/SplunkOutputTab';
import TestPreviewTab from './tabs/TestPreviewTab';

// ── Styled wrappers ──

const WizardContainer = styled.div`
    max-width: 960px;
    margin: 0 auto;
    padding: ${variables.spacingLarge};
`;

const BreadcrumbRow = styled.div`
    margin-bottom: ${variables.spacingMedium};
`;

const StepBarWrapper = styled.div`
    margin-bottom: ${variables.spacingSmall};
`;

const StepDescription = styled.div`
    display: flex;
    align-items: center;
    gap: ${variables.spacingSmall};
    padding: ${variables.spacingSmall} ${variables.spacingMedium};
    margin-bottom: ${variables.spacingMedium};
    color: ${variables.contentColorMuted};
    font-size: ${variables.fontSizeSmall};
    background: ${variables.backgroundColorPage};
    border: 1px solid ${variables.borderColor};
    border-radius: ${variables.borderRadius};
`;

const StepIcon = styled.span`
    color: ${variables.infoColor};
    font-size: 18px;
    display: inline-flex;
    flex-shrink: 0;
`;

const StepContent = styled.div`
    min-height: 300px;
    padding: ${variables.spacingSmall} 0;
`;

const WizardFooter = styled.div`
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: ${variables.spacingMedium} 0;
    margin-top: ${variables.spacingLarge};
    border-top: 1px solid ${variables.borderColor};
`;

const FooterRight = styled.div`
    display: flex;
    gap: ${variables.spacingSmall};
`;

const CenteredSpinner = styled.div`
    display: flex;
    justify-content: center;
    padding: ${variables.spacingXXLarge};
`;

// ── Inner component (needs BuilderProvider above it) ──

function StreamBuilderInner({ inputName }: { inputName?: string }) {
    const { state, dispatch } = useBuilder();
    const [activeStep, setActiveStep] = useState(0);
    const [loading, setLoading] = useState(!!inputName);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [stepError, setStepError] = useState<string | null>(null);

    const isEditing = state.mode === 'edit';

    useEffect(() => {
        if (!inputName) return;
        let cancelled = false;

        (async () => {
            try {
                const input = await getInput(inputName);
                if (cancelled) return;
                dispatch({
                    type: 'LOAD_INPUT',
                    input: { ...input, parsedManifest: {} },
                });
            } catch (err: any) {
                if (!cancelled) setError(err.message || 'Failed to load input');
            } finally {
                if (!cancelled) setLoading(false);
            }
        })();

        return () => { cancelled = true; };
    }, [inputName, dispatch]);

    const handleNext = useCallback(() => {
        setStepError(null);
        if (activeStep === 0 && !state.baseUrl) {
            setStepError('Enter a Base URL before continuing.');
            return;
        }
        setActiveStep((s) => Math.min(s + 1, STEPS.length - 1));
    }, [activeStep, state.baseUrl]);

    const handleBack = useCallback(() => {
        setStepError(null);
        setActiveStep((s) => Math.max(s - 1, 0));
    }, []);

    const handleSave = useCallback(async () => {
        setSaving(true);
        setError(null);

        try {
            const manifest = formStateToManifest(state);
            const payload: InputPayload = {
                name: state.inputName || state.streams[0]?.name || 'default',
                account: state.account,
                base_url: state.baseUrl,
                manifest,
                interval: String(state.interval),
                index: state.index,
                sourcetype: state.sourcetype,
                tags: state.tags,
                debug: state.debug ? '1' : '0',
            };

            if (isEditing && state.inputName) {
                await updateInput(state.inputName, { ...payload, disabled: state.enabled ? '0' : '1' });
            } else {
                await createInput(payload);
                if (!state.enabled) {
                    await toggleInput(payload.name, true);
                }
            }

            window.location.href = 'streams';
        } catch (err: any) {
            setError(err.message || 'Save failed');
        } finally {
            setSaving(false);
        }
    }, [state, isEditing]);

    const handleCancel = useCallback(() => {
        window.location.href = 'streams';
    }, []);

    if (loading) {
        return (
            <CenteredSpinner>
                <WaitSpinner size="large" />
            </CenteredSpinner>
        );
    }

    const streamName = state.inputName || state.streams[0]?.name || 'New Stream';

    return (
        <WizardContainer>
            <BreadcrumbRow>
                <Breadcrumbs>
                    <Breadcrumbs.Item label="Streams" to="streams" />
                    <Breadcrumbs.Item label={isEditing ? `Edit: ${streamName}` : 'New Stream'} to="#" />
                </Breadcrumbs>
            </BreadcrumbRow>

            <StepBarWrapper>
                <StepBar activeStepId={activeStep}>
                    {STEPS.map((step, idx) => (
                        <StepBar.Step key={step.label} stepId={idx}>
                            {step.label}
                        </StepBar.Step>
                    ))}
                </StepBar>
            </StepBarWrapper>

            <StepDescription>
                <StepIcon>{STEPS[activeStep].icon}</StepIcon>
                {STEPS[activeStep].description}
            </StepDescription>

            {error && (
                <Message type="error" onRequestRemove={() => setError(null)}>
                    {error}
                </Message>
            )}
            {stepError && (
                <Message type="warning" onRequestRemove={() => setStepError(null)}>
                    {stepError}
                </Message>
            )}

            <StepContent>
                {activeStep === 0 && <ConnectionTab />}
                {activeStep === 1 && <DataMappingTab />}
                {activeStep === 2 && <SplunkOutputTab />}
                {activeStep === 3 && <TestPreviewTab />}
            </StepContent>

            <WizardFooter>
                <div>
                    {activeStep === 0 ? (
                        <Button label="Cancel" onClick={handleCancel} />
                    ) : (
                        <Button
                            icon={<ChevronLeft />}
                            label="Back"
                            onClick={handleBack}
                        />
                    )}
                </div>
                <FooterRight>
                    {activeStep < STEPS.length - 1 ? (
                        <Button
                            appearance="primary"
                            label="Next"
                            onClick={handleNext}
                        />
                    ) : (
                        <>
                            <Button
                                label={saving ? 'Saving...' : 'Save'}
                                appearance="primary"
                                onClick={handleSave}
                                disabled={saving}
                            />
                        </>
                    )}
                </FooterRight>
            </WizardFooter>
        </WizardContainer>
    );
}

// ── Public component ──

export interface StreamBuilderProps {
    inputName?: string;
}

export function StreamBuilder({ inputName }: StreamBuilderProps) {
    return (
        <BuilderProvider>
            <StreamBuilderInner inputName={inputName} />
        </BuilderProvider>
    );
}

export default StreamBuilder;
