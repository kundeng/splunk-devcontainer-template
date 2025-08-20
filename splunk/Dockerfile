ARG SPLUNK_VERSION=9.2.1

#####################################################################
# MINIMAL IMAGE - defer app installation to compose/.env
#####################################################################
FROM splunk/splunk:${SPLUNK_VERSION}

# Install ACLs so Ansible can avoid world-readable temporary files
USER root
# splunk/splunk base uses Red Hat UBI (microdnf). Install via microdnf and clean up.
RUN microdnf install -y acl \
    && microdnf clean all

# Use the default entrypoint from the base image; run as ansible user
USER ansible
ENTRYPOINT ["/sbin/entrypoint.sh"]
CMD ["start-service"]