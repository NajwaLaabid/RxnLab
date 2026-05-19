# Rahti infrastructure

Kubernetes/OpenShift manifests for the RxnLab feedback persistence layer on CSC Rahti, plus operational notes for deploying the Flask app itself.

| File | Purpose |
|---|---|
| `postgres.yaml` | PVC + Service + Deployment for an in-namespace Postgres 15 (the chemist-feedback DB). |
| `postgres-backup.yaml` | Daily `pg_dump` → CSC Allas CronJob. References the image built from `backup-image/`. |
| `backup-image/Dockerfile` | Minimal Alpine + `aws-cli` + `postgresql-client` image used by the backup CronJob. Built into Rahti's internal registry; OpenShift's restricted SCC prevents installing tools at runtime, so they must be baked in. |
| `setup.sh` | First-time bootstrap: generates the Postgres password, creates both Secrets, applies the YAMLs. |

## Deploying the app

The Flask app is built by BuildConfig `retrosynthesis-lab` from `main` on the GitHub repo. A GitHub webhook auto-triggers a build on every push to `main`, so the normal flow is:

1. `git push origin main`
2. Watch the build: `oc logs -f bc/retrosynthesis-lab` (or `oc get builds`)
3. Confirm rollout: `oc rollout status deployment/retrosynthesis-lab`
4. Smoke-check: `curl -sS https://retrosynthesis-lab-diffalign.2.rahtiapp.fi/`

If the webhook didn't fire, trigger the build manually: `oc start-build retrosynthesis-lab --follow`.

### If the new pod doesn't roll over

The Deployment has an `image.openshift.io/triggers` annotation that should sha-pin the container image and roll out a new pod whenever `retrosynthesis-lab:latest` updates. In practice this hasn't been reliable — sometimes the build completes but the pod stays on the old image. Force the rollout:

```bash
oc rollout restart deployment/retrosynthesis-lab
oc rollout status deployment/retrosynthesis-lab
```

`imagePullPolicy: Always` on the container means the restart pulls the new `:latest` from the internal registry. (Note: `oc set triggers` doesn't apply here — it only works on the deprecated `DeploymentConfig`, not on plain `Deployment`.)

## First-time setup

```bash
oc login --token=... --server=https://api.2.rahti.csc.fi:6443
oc project <your-project>
bash rahti/setup.sh
oc set env deployment/<app-name> --from=secret/rxnlab-db
```

The app starts in DB-disabled mode if `DATABASE_URL` is unset, so the order above is safe: app runs degraded → DB comes up → `oc set env` triggers a rollout where the new pod connects and creates the schema.

## Enabling the Allas backup

`postgres-backup.yaml` references an image at `image-registry.openshift-image-registry.svc:5000/<namespace>/rxnlab-backup-image:latest`. The image source lives in `backup-image/Dockerfile`. **The image must exist in your namespace's registry before applying `postgres-backup.yaml`.**

### First-time image build (one-off per project)

```bash
oc new-build --name=rxnlab-backup-image --binary --strategy=docker
oc start-build rxnlab-backup-image --from-dir=rahti/backup-image --follow
```

If the build pod gets stuck in `New` with `exceeded quota`, lower its CPU:
```bash
oc patch bc/rxnlab-backup-image --type=json -p \
  '[{"op":"replace","path":"/spec/resources","value":{"limits":{"cpu":"200m","memory":"512Mi"},"requests":{"cpu":"50m","memory":"256Mi"}}}]'
```

### Re-building after updating the Dockerfile

```bash
oc start-build rxnlab-backup-image --from-dir=rahti/backup-image --follow
```

### Steps to turn on the backup CronJob:

### 1. Provision an Allas project + bucket

Allas is CSC's S3-compatible object store. You need:

1. An Allas-enabled CSC project (request via MyCSC if you don't already have one).
2. A bucket — created via the `allas-conf` CLI or rclone. See https://docs.csc.fi/data/Allas/.
3. S3 credentials for the project — see https://docs.csc.fi/data/Allas/using_allas/s3_client/.

You'll end up with:
- `AWS_ACCESS_KEY_ID` (Allas access key — short hex string)
- `AWS_SECRET_ACCESS_KEY` (Allas secret key)
- `AWS_S3_ENDPOINT` = `https://a3s.fi`
- `ALLAS_BUCKET` = the bucket name you created

### 2. Create the Rahti Secret

```bash
oc create secret generic rxnlab-allas \
  --from-literal=AWS_ACCESS_KEY_ID='<your-key>' \
  --from-literal=AWS_SECRET_ACCESS_KEY='<your-secret>' \
  --from-literal=AWS_S3_ENDPOINT='https://a3s.fi' \
  --from-literal=ALLAS_BUCKET='<your-bucket>'
```

### 3. Unsuspend the CronJob

```bash
oc patch cronjob/rxnlab-pg-backup -p '{"spec":{"suspend":false}}'
```

### 4. Verify with a one-off run

```bash
oc create job --from=cronjob/rxnlab-pg-backup test-backup
oc logs -f job/test-backup
```

Expected output ends with `done.`. Then check Allas:

```bash
oc run --rm -it allas-check --image=alpine:3.18 --restart=Never -- sh -c '
  apk add --no-cache aws-cli >/dev/null
  aws --endpoint-url=$AWS_S3_ENDPOINT s3 ls s3://$ALLAS_BUCKET/postgres/
' --env=AWS_ACCESS_KEY_ID --env=AWS_SECRET_ACCESS_KEY \
   --env=AWS_S3_ENDPOINT --env=ALLAS_BUCKET --envfrom=secret/rxnlab-allas
```

## Operational notes

- **The Postgres has no automated failover.** It's a single replica on an RWO PVC; if the node dies, the pod reschedules to another node and re-attaches the volume — typically a few-minute outage. For an HA setup, switch to actual CSC Pukki (via MyCSC) and just point `DATABASE_URL` at it.
- **The backup CronJob rotates dumps older than 30 days** in the same Allas prefix. Adjust the `30 days ago` in `postgres-backup.yaml` if you want a longer retention window.
- **Restoring from a backup:**
  ```bash
  aws --endpoint-url=https://a3s.fi s3 cp s3://<bucket>/postgres/rxnlab-YYYYMMDD-HHMMSS.sql.gz - \
    | gunzip \
    | oc exec -i deploy/rxnlab-pg -- psql -U rxnlab -d rxnlab
  ```
- **Reading the DB password locally** (if you need it for debugging):
  ```bash
  oc get secret rxnlab-pg -o jsonpath='{.data.database-password}' | base64 -d; echo
  ```
