ARG SPLUNK_VERSION=9.4.0

#####################################################################
# Custom Splunk image with skip-provision entrypoint wrapper
#
# On first start: delegates to stock entrypoint (full Ansible provision)
# On restart:     skips Ansible, starts splunkd directly (~15s vs 90s)
# Force reprovision: set FORCE_PROVISION=true or remove marker
#####################################################################
FROM splunk/splunk:${SPLUNK_VERSION}

USER root

# ACLs suppress Ansible world-readable temp file warnings
RUN microdnf install -y acl && microdnf clean all

# Entrypoint wrapper: skip Ansible on subsequent starts
COPY entrypoint-wrapper.sh /sbin/entrypoint-wrapper.sh
RUN chmod 755 /sbin/entrypoint-wrapper.sh

USER ansible
ENTRYPOINT ["/sbin/entrypoint-wrapper.sh"]
CMD ["start-service"]