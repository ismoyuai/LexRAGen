# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Dict


class RecallEvaluator:
    """召回率评估器"""

    def __init__(self, retriever, reranker):
        self.retriever = retriever
        self.reranker = reranker

    def calculate_recall(self, retrieved_nodes: List, relevant_ids: List) -> float:
        """计算单个问题的召回率"""
        retrieved_ids = [n.node.metadata["full_title"] for n in retrieved_nodes]
        hit = len(set(retrieved_ids) & set(relevant_ids))
        return hit / len(relevant_ids) if relevant_ids else 0.0

    def evaluate(self, benchmark: List[Dict]) -> float:
        """批量评估召回率"""
        results = []
        for case in benchmark:
            # 初始检索
            initial_nodes = self.retriever.retrieve(case["question"])
            # 重排序
            reranked_nodes = self.reranker.postprocess_nodes(
                initial_nodes,
                query_str=case["question"]
            )
            # 计算召回率
            recall = self.calculate_recall(reranked_nodes, case["relevant_ids"])
            results.append(recall)

            print(f"问题：{case['question']}")
            print(f"初始检索结果：{[n.node.metadata['full_title'] for n in initial_nodes]}")
            print(f"重排序后结果：{[n.node.metadata['full_title'] for n in reranked_nodes]}")
            print(f"召回条款：{[n.node.metadata['full_title'] for n in reranked_nodes[:3]]}")
            print(f"目标条款：{case['relevant_ids']}")
            print(f"召回率：{recall:.1%}\n")

            # print(f"\n问题：{case['question']}")
            # print(f"目标条款：{case['relevant_ids']}")
            # print(f"召回率：{recall:.1%}")
        avg_recall = np.mean(results)
        print(f"平均召回率：{avg_recall:.1%}")
        return avg_recall

class E2EEvaluator:
    def __init__(self, query_engine):
        self.query_engine = query_engine

    def evaluate_case(self, response, standard):
        try:
            # 获取实际命中的条款
            retrieved_clauses = [node.node.metadata["full_title"] for node in response.source_nodes]

            # 获取标准答案要求的条款
            required_clauses = standard["standard_answer"]["条款"]

            # 计算命中情况
            hit_clauses = list(set(retrieved_clauses) & set(required_clauses))
            missed_clauses = list(set(required_clauses) - set(retrieved_clauses))

            # 计算命中率
            clause_hit = len(hit_clauses) / len(required_clauses) if required_clauses else 0.0

            return {
                "clause_score": clause_hit,
                "hit_clauses": hit_clauses,
                "missed_clauses": missed_clauses
            }
        except Exception as e:
            print(f"评估失败：{str(e)}")
            return None

    def evaluate(self, benchmark):
        results = []
        for case in benchmark:
            try:
                response = self.query_engine.query(case["question"])
                case_result = self.evaluate_case(response, case)

                if case_result:
                    print(f"\n问题：{case['question']}")
                    print(f"命中条款：{case_result['hit_clauses']}")
                    print(f"缺失条款：{case_result['missed_clauses']}")
                    print(f"条款命中率：{case_result['clause_score']:.1%}")
                    results.append(case_result)
                else:
                    results.append(None)
            except Exception as e:
                print(f"查询失败：{str(e)}")
                results.append(None)

        # 计算统计数据
        valid_results = [r for r in results if r is not None]
        avg_hit = np.mean([r["clause_score"] for r in valid_results]) if valid_results else 0

        print("\n=== 最终评估报告 ===")
        print(f"有效评估案例：{len(valid_results)}/{len(benchmark)}")
        print(f"平均条款命中率：{avg_hit:.1%}")

        # 输出详细错误分析
        for i, result in enumerate(results):
            if result is None:
                print(f"案例{i + 1}：{benchmark[i]['question']} 评估失败")

        return results