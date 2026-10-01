pub struct ConsistentHasher {
    nodes: Vec<String>,
    #[allow(dead_code)]
    vnodes: usize,
}

impl ConsistentHasher {
    pub fn new(mut nodes: Vec<String>, vnodes: usize) -> Self {
        nodes.retain(|node| !node.is_empty());
        nodes.sort();
        nodes.dedup();
        Self {
            nodes,
            vnodes: vnodes.clamp(1, 1024),
        }
    }

    pub fn get_shard(&self, key: &str) -> String {
        if self.nodes.is_empty() {
            return "local".to_string();
        }

        let mut best_node = self.nodes[0].clone();
        let mut best_score = u64::MIN;

        for node in &self.nodes {
            let replicas = self.vnodes.max(1);
            for vnode in 0..replicas {
                let score = stable_score(key, node, vnode);
                if score > best_score || (score == best_score && node < &best_node) {
                    best_score = score;
                    best_node = node.clone();
                }
            }
        }

        best_node
    }
}

fn stable_score(key: &str, node: &str, vnode: usize) -> u64 {
    let mut hash = 1469598103934665603u64;
    for byte in key
        .as_bytes()
        .iter()
        .copied()
        .chain([0xff])
        .chain(node.as_bytes().iter().copied())
        .chain([0xfe])
        .chain((vnode as u64).to_le_bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(1099511628211u64);
    }
    // SplitMix64 avalanche makes the rendezvous score uniform while retaining
    // a fully specified cross-process/cross-version hash contract.
    let mut value = hash;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::ConsistentHasher;

    #[test]
    fn rendezvous_hashing_is_stable_for_same_key() {
        let hasher = ConsistentHasher::new(
            vec![
                "node-a".to_string(),
                "node-b".to_string(),
                "node-c".to_string(),
            ],
            8,
        );
        let first = hasher.get_shard("session-42");
        let second = hasher.get_shard("session-42");
        assert_eq!(first, second);
    }

    #[test]
    fn empty_cluster_falls_back_to_local() {
        let hasher = ConsistentHasher::new(Vec::new(), 4);
        assert_eq!(hasher.get_shard("any"), "local");
    }
}
