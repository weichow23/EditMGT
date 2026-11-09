"""Read-only strict Teacher/data validation before allocating all four roles."""
import argparse,json
from src.dimo.config import resolve_config,validate_config,source_metadata
from src.dimo.origin import select_teacher
from src.dimo.infer import validate_base_components
from src.dimo.initialization import parent_metadata,parent_teacher


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    args=p.parse_args()
    c=resolve_config(args.config)
    validate_config(c)
    validate_base_components(c['base_model_dir'])
    parent_metadata(c)
    model,origin=parent_teacher(c) if c.get('initialization')=='finetune' else select_teacher(c)
    if c.get('data_manifest'):
        from src.dimo.manifest_data import ManifestDataset
        # Structural/path validation is independent of text encoder allocation.
        dataset=ManifestDataset(c,None,0,1)
        data=dict(samples=len(dataset.rows),manifest_sha256=c['data_manifest_sha256'],
                  counts=c['expected_dataset_counts'],image_decode='checked when consuming each sample; not a train/test overlap audit')
    else:data=dict(mode='local_stream',roots_checked=True)
    print(json.dumps(dict(teacher=origin,data=data,source_metadata=source_metadata(),
        source_halo=model.config.sparse_target_halo,source_supervise_halo=model.config.supervise_target_halo,
        runtime_halo=c.get('training_sparse_target_halo',model.config.sparse_target_halo),
        runtime_supervise_halo=c.get('training_supervise_target_halo',model.config.supervise_target_halo)),indent=2))


if __name__=='__main__':main()
