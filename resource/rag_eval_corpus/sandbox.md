# Sandbox and platform execution

Poirot has Local and Docker Sandbox providers. Local Sandbox is convenient for
development; Docker Sandbox isolates command execution from the host.

DockerPathTranslator converts container paths back to Windows host paths for
artifact extraction. DockerPathGuard restricts writes to the mounted
/mnt/poirot/user-data/ area. A warm pool reduces container cold-start latency.
The WslDockerExecutor translates a Windows path such as D:\\foo\\bar to
/mnt/d/foo/bar for a Docker daemon running in WSL2.
