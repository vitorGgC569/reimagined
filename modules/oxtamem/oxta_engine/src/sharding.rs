use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct ConsistentHasher {
    nodes: Vec<String>,
    #[allow(dead_code)]
    vnodes: usize,
}

impl ConsistentHasher {
    pub fn new(nodes: Vec<String>, vnodes: usize) -> Self {
        Self { nodes, vnodes }
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
                let mut hasher = DefaultHasher::new();
                key.hash(&mut hasher);
                node.hash(&mut hasher);
                vnode.hash(&mut hasher);
                let score = hasher.finish();
                if score > best_score {
                    best_score = score;
                    best_node = node.clone();
                }
            }
        }

        best_node
    }
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
