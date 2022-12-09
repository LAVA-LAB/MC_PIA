mkdir -m777 mlruns
mkdir -m777 mlruns/0
mkdir -m777 mlruns/.trash
echo -e "artifact_location: ./mlruns/0\nexperiment_id: '0'\nlifecycle_stage: active\nname: Default" > mlruns/0/meta.yaml
chmod 777 mlruns/0/meta.yaml