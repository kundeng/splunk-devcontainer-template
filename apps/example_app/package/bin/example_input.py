"""Example modular input handler — replace with your collection logic."""

import import_declare_test  # noqa: F401 — must be first; sets up sys.path
import json
import sys

from splunktaucclib.modinput_wrapper.base_modinput import BaseModInput


class ExampleInput(BaseModInput):
    app = "example_app"
    name = "example_input"
    use_single_instance = False

    def get_scheme(self):
        scheme = super().get_scheme()
        scheme.title = "Example Input"
        scheme.description = "Collects data from an example API"
        return scheme

    def stream_events(self, inputs, ew):
        for input_name, input_item in inputs.inputs.items():
            self.log_info(f"Starting collection for {input_name}")

            # Your collection logic here
            record = {
                "message": "Hello from the example input!",
                "input_name": input_name,
            }

            event = self._create_event(
                data=json.dumps(record),
                index=input_item.get("index", "default"),
                sourcetype="example:json",
            )
            ew.write_event(event)

    def _create_event(self, data, index, sourcetype):
        from splunklib.modularinput import Event

        event = Event()
        event.data = data
        event.index = index
        event.sourcetype = sourcetype
        return event


if __name__ == "__main__":
    exit_code = ExampleInput().run(sys.argv)
    sys.exit(exit_code)
