def __getattr__(name):
    _imports = {
        "ConnectorBuilderFileUploader": ".connector_builder_file_uploader",
        "DefaultFileUploader": ".default_file_uploader",
        "FileUploader": ".file_uploader",
        "FileWriter": ".file_writer",
        "LocalFileSystemFileWriter": ".local_file_system_file_writer",
        "NoopFileWriter": ".noop_file_writer",
    }
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name], package=__name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
