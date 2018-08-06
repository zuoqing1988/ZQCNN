#ifndef _ZQ_TRIANGLE_MESH_MEDIAL_AXIS_H_
#define _ZQ_TRIANGLE_MESH_MEDIAL_AXIS_H_
#pragma once

#include <vector>
#include "ZQ_Vec2D.h"

namespace ZQ
{
	class ZQ_TriangleMeshMedialAxis
	{
	private:
		class Edge
		{
		public:
			int v_id[2];
			int appear_count;

			Edge(int id0 = -1, int id1 = -1) 
			{ 
				appear_count = 1; 
				v_id[0] = id0;
				v_id[1] = id1;
			}
			bool SameEdge(const Edge& e) const
			{
				return (v_id[0] == e.v_id[0] && v_id[1] == e.v_id[1])
					|| (v_id[0] == e.v_id[1] && v_id[1] == e.v_id[0]);
			}
		};

		class Triangle
		{
		public:
			Edge e[3];
		};

	public:

		class ListNode
		{
		public:
			bool isHead;
			int triangle_id;
			int v_id;
			int edge_v_id[2];
			bool isEnd;
		};

		class JointNode
		{
		public:
			int triangle_id;
			int edge0_v_id[2];
			int edge1_v_id[2];
			int edge2_v_id[2];
		};

		class Line
		{
		public:
			ZQ_Vec2D p[2];
		};
	
	public:
		static bool ExtractMedialAxis(const std::vector<int>& indices, std::vector<std::vector<ListNode>>& lists, std::vector<JointNode>& joints)
		{
			lists.clear();
			joints.clear();
			int indices_size = indices.size();
			if (indices_size % 3 != 0)
				return false;
			int triangle_num = indices_size / 3;
			if (triangle_num == 0)
			{
				printf("no triangles\n");
				return false;
			}
			if (triangle_num == 1)
			{
				printf("only one triangle\n");
				return false;
			}

			std::vector<Edge> edges;
			std::vector<Triangle> triangles(triangle_num);
			std::vector<int> T_triangle_ids;	//who has 1 neighbor
			std::vector<int> S_triangle_ids;	//who has 2 neighbors
			std::vector<int> J_triangle_ids;	//who has 3 neighbors

			for (int i = 0; i < triangle_num; i++)
			{
				Edge edge01(indices[i * 3 + 0], indices[i * 3 + 1]);
				Edge edge12(indices[i * 3 + 1], indices[i * 3 + 2]);
				Edge edge20(indices[i * 3 + 2], indices[i * 3 + 0]);
				triangles[i].e[0] = edge01;
				triangles[i].e[1] = edge12;
				triangles[i].e[2] = edge20;
				int pos = _find_edge(edges, edge01);
				if (pos >= 0)
					edges[pos].appear_count++;
				else
					edges.push_back(edge01);
				pos = _find_edge(edges, edge12);
				if (pos >= 0)
					edges[pos].appear_count++;
				else
					edges.push_back(edge12);
				pos = _find_edge(edges, edge20);
				if (pos >= 0)
					edges[pos].appear_count++;
				else
					edges.push_back(edge20);
			}

			for (int i = 0; i < triangle_num; i++)
			{
				Edge edge01(indices[i * 3 + 0], indices[i * 3 + 1]);
				Edge edge12(indices[i * 3 + 1], indices[i * 3 + 2]);
				Edge edge20(indices[i * 3 + 2], indices[i * 3 + 0]);
				int pos01 = _find_edge(edges, edge01);
				int pos12 = _find_edge(edges, edge12);
				int pos20 = _find_edge(edges, edge20);
				int count = 0;
				if (edges[pos01].appear_count == 2)
					count++;
				if (edges[pos12].appear_count == 2)
					count++;
				if (edges[pos20].appear_count == 2)
					count++;

				if (count == 3)
					J_triangle_ids.push_back(i);
				else if (count == 2)
					S_triangle_ids.push_back(i);
				else
					T_triangle_ids.push_back(i);
			}

			/**************************************/

			//prepare flags
			std::vector<int> tri_flag(triangle_num);
			std::vector<int> tri_used_flag(triangle_num);
			memset(&tri_used_flag[0], 0, sizeof(int)*triangle_num);
			for (int i = 0; i < T_triangle_ids.size(); i++)
			{
				tri_flag[T_triangle_ids[i]] = 1;
			}
			for (int i = 0; i < S_triangle_ids.size(); i++)
			{
				tri_flag[S_triangle_ids[i]] = 2;
			}
			for (int i = 0; i < J_triangle_ids.size(); i++)
			{
				tri_flag[J_triangle_ids[i]] = 3;
			}

			//handle from T traingles
			for (int i = 0; i < T_triangle_ids.size(); i++)
			{
				int tri_id = T_triangle_ids[i];
				int id0 = indices[tri_id * 3 + 0];
				int id1 = indices[tri_id * 3 + 1];
				int id2 = indices[tri_id * 3 + 2];
				
				int corner_id = -1, edge_id0 = -1, edge_id1 = -1;
				Edge edge01(id0, id1), edge12(id1, id2), edge20(id2, id0);
				
				if (edges[_find_edge(edges, edge01)].appear_count == 2)
				{
					corner_id = id2;
					edge_id0 = id0;
					edge_id1 = id1;
				}
				else if (edges[_find_edge(edges, edge12)].appear_count == 2)
				{
					corner_id = id0;
					edge_id0 = id1;
					edge_id1 = id2;
				}
				else if (edges[_find_edge(edges, edge20)].appear_count == 2)
				{
					corner_id = id1;
					edge_id0 = id2;
					edge_id1 = id0;
				}
				else
				{
					printf("may be the triangle mesh is invalid\n");
					return false;
				}
				
				std::vector<ListNode> cat_list;
				ListNode node;
				node.isHead = true;
				node.isEnd = false;
				node.v_id = corner_id;
				node.edge_v_id[0] = edge_id0;
				node.edge_v_id[1] = edge_id1;
				node.triangle_id = tri_id;

				cat_list.push_back(node);

				tri_used_flag[tri_id] = 1;

				//trace list
				Edge search_edge(edge_id0, edge_id1);
				int last_tri_id = tri_id;
				while (true)
				{
					std::vector<int> tmp_tri_id;
					_find_triangle(triangles, search_edge, tmp_tri_id);
					if (tmp_tri_id.size() != 2)
					{
						printf("may be the triangle mesh is invalid\n");
						return false;
					}

					int cur_tri_id = tmp_tri_id[0] == last_tri_id ? tmp_tri_id[1] : tmp_tri_id[0];
					
					if (tri_flag[cur_tri_id] == 1)
					{
						break;
						
					}


					if (tri_flag[cur_tri_id] == 2)
					{
						int cur_pt_id0 = indices[cur_tri_id * 3 + 0];
						int cur_pt_id1 = indices[cur_tri_id * 3 + 1];
						int cur_pt_id2 = indices[cur_tri_id * 3 + 2];
						
						Edge edge01(cur_pt_id0,cur_pt_id1), edge12(cur_pt_id1,cur_pt_id2), edge20(cur_pt_id2,cur_pt_id0);
						
						if (edge01.SameEdge(search_edge))
						{
							if (edges[_find_edge(edges, edge20)].appear_count == 1)
								search_edge = edge12;
							else
								search_edge = edge20;
						}
						else if (edge20.SameEdge(search_edge))
						{
							if (edges[_find_edge(edges, edge01)].appear_count == 1)
								search_edge = edge12;
							else
								search_edge = edge01;
						}
						else
						{
							if (edges[_find_edge(edges, edge20)].appear_count == 1)
								search_edge = edge01;
							else
								search_edge = edge20;
						}

						ListNode cur_node;
						cur_node.isHead = false;
						cur_node.isEnd = false;
						cur_node.edge_v_id[0] = search_edge.v_id[0];
						cur_node.edge_v_id[1] = search_edge.v_id[1];
						cur_node.triangle_id = cur_tri_id;

						cat_list.push_back(cur_node);

						tri_used_flag[cur_tri_id] = 1;
						last_tri_id = cur_tri_id;
					}
					else //tri_flag[] == 3
					{
						int len = cat_list.size();
						cat_list[len - 1].isEnd = true;
						break;
					}
				}
				lists.push_back(cat_list);
			}

			//handle J triangle
			for (int i = 0; i < J_triangle_ids.size(); i++)
			{
				int tri_id = J_triangle_ids[i];
				
				JointNode node;
				node.triangle_id = tri_id;
				node.edge0_v_id[0] = indices[tri_id * 3 + 0];
				node.edge0_v_id[1] = indices[tri_id * 3 + 1];
				node.edge1_v_id[0] = indices[tri_id * 3 + 1];
				node.edge1_v_id[1] = indices[tri_id * 3 + 2];
				node.edge2_v_id[0] = indices[tri_id * 3 + 2];
				node.edge2_v_id[1] = indices[tri_id * 3 + 0];

				joints.push_back(node);
				tri_used_flag[tri_id] = 1;

				for (int kk = 0; kk < 3; kk++)
				{
					//handle remained S triangle
					std::vector<ListNode> cat_list;
					ListNode listnode;
					listnode.isHead = false;
					listnode.isEnd = false;
					if (kk == 0)
					{
						listnode.edge_v_id[0] = node.edge0_v_id[0];
						listnode.edge_v_id[1] = node.edge0_v_id[1];
					}
					else if (kk == 1)
					{
						listnode.edge_v_id[0] = node.edge1_v_id[0];
						listnode.edge_v_id[1] = node.edge1_v_id[1];
					}
					else if (kk == 2)
					{
						listnode.edge_v_id[0] = node.edge2_v_id[0];
						listnode.edge_v_id[1] = node.edge2_v_id[1];
					}
					listnode.triangle_id = tri_id;
					cat_list.push_back(listnode);

					//trace list
					Edge search_edge;
					search_edge.v_id[0] = listnode.edge_v_id[0];
					search_edge.v_id[1] = listnode.edge_v_id[1];
					int last_tri_id = tri_id;
					while (true)
					{
						std::vector<int> tmp_tri_id;
						_find_triangle(triangles, search_edge, tmp_tri_id);
						if (tmp_tri_id.size() != 2)
						{
							printf("may be the triangle mesh is invalid\n");
							return false;
						}

						int cur_tri_id = tmp_tri_id[0];
						if (cur_tri_id == last_tri_id)
							cur_tri_id = tmp_tri_id[1];

						if (tri_used_flag[cur_tri_id] == true)
						{
							break;
						}

						if (tri_flag[cur_tri_id] == 2)
						{
							Edge edge01(indices[cur_tri_id * 3 + 0], indices[cur_tri_id * 3 + 1]);
							Edge edge12(indices[cur_tri_id * 3 + 1], indices[cur_tri_id * 3 + 2]);
							Edge edge02(indices[cur_tri_id * 3 + 0], indices[cur_tri_id * 3 + 2]);
							
							if (edge01.SameEdge(search_edge))
							{
								if (edges[_find_edge(edges, edge02)].appear_count == 1)
									search_edge = edge12;
								else
									search_edge = edge02;
							}
							else if (edge02.SameEdge(search_edge))
							{
								if (edges[_find_edge(edges, edge12)].appear_count == 1)
									search_edge = edge01;
								else
									search_edge = edge12;
							}
							else
							{
								if (edges[_find_edge(edges, edge02)].appear_count == 1)
									search_edge = edge01;
								else
									search_edge = edge02;
							}

							ListNode cur_node;
							cur_node.isHead = false;
							cur_node.isEnd = false;
							cur_node.edge_v_id[0] = search_edge.v_id[0];
							cur_node.edge_v_id[1] = search_edge.v_id[1];
							cur_node.triangle_id = cur_tri_id;

							cat_list.push_back(cur_node);

							tri_used_flag[cur_tri_id] = 1;
							last_tri_id = cur_tri_id;
						}
						else //tri_flag[] == 3
						{
							int len = cat_list.size();
							cat_list[len - 1].isEnd = true;
							break;
						}
					}
					if (cat_list.size() > 1)
						lists.push_back(cat_list);

				}
			}


			//check whether all triangles have been used

			int check_all_used_flag = true;
			for (int i = 0; i < triangle_num; i++)
			{
				if (tri_used_flag[i] == 0)
				{
					check_all_used_flag = false;
					break;
				}
			}
			if (check_all_used_flag == false)
			{
				printf("wrong:not all triangles have been used, may be the triangle mesh is invalid\n");
				return false;
			}
			return true;
		}

		static bool TranslateToLines(const std::vector<ZQ_Vec2D>& pts, const std::vector<int>& indices,
			const std::vector<std::vector<ListNode>>& lists, const std::vector<JointNode>& joints, std::vector<Line>& lines)
		{
			for (int i = 0; i < lists.size(); i++)
			{
				if (lists[i].size() == 0)
					continue;
				if (!((lists[i][0].isHead == true) || (lists[i].size() >= 2)))
				{
					printf("something is wrong\n");
					lines.clear();
					return false;
				}

				ZQ_Vec2D p0;
				int j;
				if (lists[i][0].isHead)
				{
					j = 0;
					int pt_id = lists[i][0].v_id;
					p0 = pts[pt_id];
				}
				else
				{
					j = 0;
					int edge_pt_id0 = lists[i][j].edge_v_id[0];
					int edge_pt_id1 = lists[i][j].edge_v_id[1];
					ZQ_Vec2D edge_p0 = pts[edge_pt_id0];
					ZQ_Vec2D edge_p1 = pts[edge_pt_id1];
					p0.x = 0.5*(edge_p0.x + edge_p1.x);
					p0.y = 0.5*(edge_p0.y + edge_p1.y);
					j++;
				}

				do {
					int edge_pt_id0 = lists[i][j].edge_v_id[0];
					int edge_pt_id1 = lists[i][j].edge_v_id[1];
					ZQ_Vec2D edge_p0 = pts[edge_pt_id0];
					ZQ_Vec2D edge_p1 = pts[edge_pt_id1];

					ZQ_Vec2D p1;
					p1.x = 0.5*(edge_p0.x + edge_p1.x);
					p1.y = 0.5*(edge_p0.y + edge_p1.y);

					Line cur_l;
					cur_l.p[0] = p0;
					cur_l.p[1] = p1;

					lines.push_back(cur_l);

					p0 = p1;
					j++;
				} while (j < lists[i].size());
			}

			for (int i = 0; i < joints.size(); i++)
			{
				int tri_id = joints[i].triangle_id;
				int pt_id0 = indices[tri_id * 3 + 0];
				int pt_id1 = indices[tri_id * 3 + 1];
				int pt_id2 = indices[tri_id * 3 + 2];
				
				ZQ_Vec2D p0 = pts[pt_id0];
				ZQ_Vec2D p1 = pts[pt_id1];
				ZQ_Vec2D p2 = pts[pt_id2];

				ZQ_Vec2D center, edge_center0, edge_center1, edge_center2;
				center.x = 1.0 / 3 * (p0.x + p1.x + p2.x);
				center.y = 1.0 / 3 * (p0.y + p1.y + p2.y);
				edge_center0.x = 0.5*(p0.x + p1.x);
				edge_center0.y = 0.5*(p0.y + p1.y);
				edge_center1.x = 0.5*(p1.x + p2.x);
				edge_center1.y = 0.5*(p1.y + p2.y);
				edge_center2.x = 0.5*(p2.x + p0.x);
				edge_center2.y = 0.5*(p2.y + p0.y);

				Line cur_l;
				cur_l.p[0] = center;
				cur_l.p[1] = edge_center0;
				lines.push_back(cur_l);
				cur_l.p[1] = edge_center1;
				lines.push_back(cur_l);
				cur_l.p[1] = edge_center2;
				lines.push_back(cur_l);
			}
			return true;
		}

	private:
		
		static int _find_edge(const std::vector<Edge>& edges, const Edge& e)
		{
			for (int i = 0; i < edges.size(); i++)
			{
				if (edges[i].SameEdge(e))
					return i;
			}
			return -1;
		}

		static void _find_triangle(const std::vector<Triangle>& triangles, const Edge& e, std::vector<int>& tri_id)
		{
			tri_id.clear();
			int triangle_num = triangles.size();
			for (int i = 0; i < triangle_num; i++)
			{
				if (e.SameEdge(triangles[i].e[0]) || e.SameEdge(triangles[i].e[1]) || e.SameEdge(triangles[i].e[2]))
				{
					tri_id.push_back(i);
				}
			}
		}
	};


}
#endif
