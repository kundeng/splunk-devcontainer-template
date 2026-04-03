import { renderHook } from '@testing-library/react';
import { useUnsavedGuard } from './useUnsavedGuard';

describe('useUnsavedGuard', () => {
    test('registers beforeunload when dirty', () => {
        const addSpy = jest.spyOn(window, 'addEventListener');
        renderHook(() => useUnsavedGuard(true));
        expect(addSpy).toHaveBeenCalledWith('beforeunload', expect.any(Function));
        addSpy.mockRestore();
    });

    test('preventDefault called when dirty', () => {
        renderHook(() => useUnsavedGuard(true));
        const event = new Event('beforeunload', { cancelable: true });
        const preventSpy = jest.spyOn(event, 'preventDefault');
        window.dispatchEvent(event);
        expect(preventSpy).toHaveBeenCalled();
    });

    test('preventDefault NOT called when clean', () => {
        renderHook(() => useUnsavedGuard(false));
        const event = new Event('beforeunload', { cancelable: true });
        const preventSpy = jest.spyOn(event, 'preventDefault');
        window.dispatchEvent(event);
        expect(preventSpy).not.toHaveBeenCalled();
    });

    test('removes listener on unmount', () => {
        const removeSpy = jest.spyOn(window, 'removeEventListener');
        const { unmount } = renderHook(() => useUnsavedGuard(true));
        unmount();
        expect(removeSpy).toHaveBeenCalledWith('beforeunload', expect.any(Function));
        removeSpy.mockRestore();
    });
});
