/**
 * ImportModal — File picker, preview, conflict resolution, and bulk create.
 */

import React, { useState, useCallback } from 'react';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import Modal from '@splunk/react-ui/Modal';
import Button from '@splunk/react-ui/Button';
import Table from '@splunk/react-ui/Table';
import Select from '@splunk/react-ui/Select';
import Badge from '@splunk/react-ui/Badge';
import Message from '@splunk/react-ui/Message';
import WaitSpinner from '@splunk/react-ui/WaitSpinner';

import { parseImportFile, detectConflicts } from '../utils/export-import';
import type { ImportedStream } from '../utils/export-import';
import type { InputSummary } from '../types';
import { createInput, updateInput } from '../services/splunk-api';

const ModalBody = styled.div`
    min-width: 600px;
    max-height: 500px;
    overflow-y: auto;
`;

const FileInput = styled.input`
    margin: ${variables.spacingMedium} 0;
`;

const Summary = styled.div`
    margin-top: ${variables.spacingMedium};
    font-size: ${variables.fontSizeSmall};
`;

interface ImportModalProps {
    open: boolean;
    onClose: () => void;
    existingInputs: InputSummary[];
    onComplete: () => void;
}

export default function ImportModal({ open, onClose, existingInputs, onComplete }: ImportModalProps) {
    const [streams, setStreams] = useState<ImportedStream[]>([]);
    const [parseError, setParseError] = useState<string | null>(null);
    const [importing, setImporting] = useState(false);
    const [result, setResult] = useState<{ created: number; skipped: number; failed: string[] } | null>(null);

    const handleFileChange = useCallback(
        (e: React.ChangeEvent<HTMLInputElement>) => {
            setParseError(null);
            setResult(null);
            const file = e.target.files?.[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = () => {
                try {
                    const content = reader.result as string;
                    const parsed = parseImportFile(content);
                    if (parsed.length === 0) {
                        setParseError('No valid stream configurations found in the file.');
                        return;
                    }
                    const { clean, conflicts } = detectConflicts(parsed, existingInputs);
                    setStreams([
                        ...clean,
                        ...conflicts,
                    ]);
                } catch (err: any) {
                    setParseError(err.message || 'Failed to parse file.');
                }
            };
            reader.readAsText(file);
        },
        [existingInputs]
    );

    const handleResolutionChange = useCallback(
        (name: string, resolution: ImportedStream['resolution']) => {
            setStreams((prev) =>
                prev.map((s) => (s.name === name ? { ...s, resolution } : s))
            );
        },
        []
    );

    const handleImport = useCallback(async () => {
        setImporting(true);
        setResult(null);

        let created = 0;
        let skipped = 0;
        const failed: string[] = [];

        for (const stream of streams) {
            if (stream.resolution === 'skip') {
                skipped++;
                continue;
            }

            const payload = {
                name: stream.resolution === 'rename' ? `${stream.name}-imported` : stream.name,
                account: stream.account,
                base_url: stream.base_url,
                manifest: stream.manifest,
                interval: String(stream.interval),
                index: stream.index,
                sourcetype: stream.sourcetype,
                tags: stream.tags,
            };

            try {
                if (stream.resolution === 'overwrite' && stream.conflict === 'name') {
                    await updateInput(stream.name, payload);
                } else {
                    await createInput(payload);
                }
                created++;
            } catch (err: any) {
                failed.push(`${stream.name}: ${err.message}`);
            }
        }

        setResult({ created, skipped, failed });
        setImporting(false);

        if (failed.length === 0) {
            onComplete();
        }
    }, [streams, onComplete]);

    const handleClose = useCallback(() => {
        setStreams([]);
        setParseError(null);
        setResult(null);
        onClose();
    }, [onClose]);

    return (
        <Modal open={open} onRequestClose={handleClose}>
            <Modal.Header title="Import Stream Configurations" onRequestClose={handleClose} />
            <Modal.Body>
                <ModalBody>
                    <FileInput
                        type="file"
                        accept=".yaml,.yml,.json"
                        onChange={handleFileChange}
                    />

                    {parseError && (
                        <Message type="error">{parseError}</Message>
                    )}

                    {streams.length > 0 && !result && (
                        <>
                            <Table stripeRows>
                                <Table.Head>
                                    <Table.HeadCell>Name</Table.HeadCell>
                                    <Table.HeadCell>Account</Table.HeadCell>
                                    <Table.HeadCell>Status</Table.HeadCell>
                                    <Table.HeadCell>Action</Table.HeadCell>
                                </Table.Head>
                                <Table.Body>
                                    {streams.map((s) => (
                                        <Table.Row key={s.name}>
                                            <Table.Cell>{s.name}</Table.Cell>
                                            <Table.Cell>{s.account || '\u2014'}</Table.Cell>
                                            <Table.Cell>
                                                <Badge
                                                    label={s.conflict === 'name' ? 'Conflict' : 'New'}
                                                    backgroundColor={s.conflict === 'name' ? '#d41f1c' : variables.accentColorPositive}
                                                    foregroundColor="#fff"
                                                />
                                            </Table.Cell>
                                            <Table.Cell>
                                                {s.conflict === 'name' ? (
                                                    <Select
                                                        value={s.resolution}
                                                        onChange={(_e: any, { value }: any) => handleResolutionChange(s.name, value)}
                                                        style={{ width: 120 }}
                                                    >
                                                        <Select.Option value="skip" label="Skip" />
                                                        <Select.Option value="rename" label="Rename" />
                                                        <Select.Option value="overwrite" label="Overwrite" />
                                                    </Select>
                                                ) : (
                                                    <span style={{ color: variables.contentColorMuted }}>Create</span>
                                                )}
                                            </Table.Cell>
                                        </Table.Row>
                                    ))}
                                </Table.Body>
                            </Table>
                        </>
                    )}

                    {result && (
                        <Summary>
                            <Message type={result.failed.length > 0 ? 'warning' : 'success'}>
                                Imported {result.created} stream{result.created !== 1 ? 's' : ''}.
                                {result.skipped > 0 && ` Skipped ${result.skipped}.`}
                                {result.failed.length > 0 && (
                                    <>
                                        <br />
                                        {result.failed.map((f, i) => (
                                            <div key={i}>{f}</div>
                                        ))}
                                    </>
                                )}
                            </Message>
                        </Summary>
                    )}
                </ModalBody>
            </Modal.Body>
            <Modal.Footer>
                {importing && <WaitSpinner />}
                <Button label="Cancel" onClick={handleClose} />
                {!result && streams.length > 0 && (
                    <Button
                        appearance="primary"
                        label={importing ? 'Importing...' : `Import ${streams.filter((s) => s.resolution !== 'skip').length} Stream(s)`}
                        onClick={handleImport}
                        disabled={importing || streams.filter((s) => s.resolution !== 'skip').length === 0}
                    />
                )}
                {result && result.failed.length === 0 && (
                    <Button appearance="primary" label="Done" onClick={handleClose} />
                )}
            </Modal.Footer>
        </Modal>
    );
}
