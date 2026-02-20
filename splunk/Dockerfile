ARG SPLUNK_VERSION=9.4.0

#####################################################################
# BASE STAGE — shared foundation for dev and staging
#####################################################################
FROM splunk/splunk:${SPLUNK_VERSION} AS base

USER root

# ACLs suppress Ansible world-readable temp file warnings
RUN microdnf install -y acl && microdnf clean all

# Entrypoint wrapper: skip Ansible on subsequent starts
COPY entrypoint-wrapper.sh /sbin/entrypoint-wrapper.sh
RUN chmod 755 /sbin/entrypoint-wrapper.sh

#####################################################################
# DEV STAGE — local development with bind-mounted apps
#
# On first start: delegates to stock entrypoint (full Ansible provision)
# On restart:     skips Ansible, starts splunkd directly (~15s vs 90s)
# Force reprovision: set FORCE_PROVISION=true or remove marker
# Apps are bind-mounted + symlinked (not baked in)
#####################################################################
FROM base AS dev
USER ansible
ENTRYPOINT ["/sbin/entrypoint-wrapper.sh"]
CMD ["start-service"]