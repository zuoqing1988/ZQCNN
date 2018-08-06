#ifndef _ZQ_SCANLINE_POLYGON_FILL_H_
#define _ZQ_SCANLINE_POLYGON_FILL_H_
#pragma once

#include <vector>
#include <math.h>
#include "ZQ_Vec2D.h"

namespace ZQ
{
	class ZQ_ScanLinePolygonFill
	{
		enum ClipMode{ CLIPMODE_BOTTOM, CLIPMODE_TOP, CLIPMODE_LEFT, CLIPMODE_RIGHT };
		class ScanLineEdge
		{
		public:
			double ymax;
			double xi;
			double dx;
		};
	public:
		ZQ_ScanLinePolygonFill() {}
		~ZQ_ScanLinePolygonFill() {}

		static bool ScanLinePolygonFill(const std::vector<ZQ_Vec2D>& polygon_pts, std::vector<ZQ_Vec2D>& pixels)
		{
			if (polygon_pts.size() < 3)
				return true;

			pixels.clear();

			int ymin = 0;
			int ymax = 0;
			_getPolygonMinMax(polygon_pts, ymin, ymax);

			std::vector<std::vector<ScanLineEdge>> slNet(ymax - ymin + 1);

			_initScanLineNewEdgeTable(polygon_pts, ymin, ymax, slNet);

			_processScanLineFill(slNet, ymin, ymax, pixels);

			return true;
		}

		static bool ScanLinePolygonFillWithClip(const std::vector<ZQ_Vec2D>& polygon_pts, int width, int height, std::vector<ZQ_Vec2D>& pixels)
		{

			std::vector<ZQ_Vec2D> out_poly;
			if (!ClipPolygon(polygon_pts, width, height, out_poly))
				return false;

			if (out_poly.size() < 3)
				return true;

			pixels.clear();

			int ymin = 0;
			int ymax = 0;
			_getPolygonMinMax(polygon_pts, ymin, ymax);

			std::vector<std::vector<ScanLineEdge>> slNet(ymax - ymin + 1);

			_initScanLineNewEdgeTable(polygon_pts, ymin, ymax, slNet);

			_processScanLineFill(slNet, ymin, ymax, pixels);

			return true;
		}

		static bool FillOneStrokeWithClip(const ZQ_Vec2D& start_pos, const ZQ_Vec2D& end_pos, float half_size, int width, int height, std::vector<ZQ_Vec2D>& pixels)
		{
			pixels.clear();

			std::vector<ZQ_Vec2D> poly;
			if (end_pos.y >= start_pos.y)
			{
				if (end_pos.x >= start_pos.x)
				{
					poly.push_back(ZQ_Vec2D(end_pos.x + half_size, end_pos.y - half_size));
					poly.push_back(ZQ_Vec2D(end_pos.x + half_size, end_pos.y + half_size));
					poly.push_back(ZQ_Vec2D(end_pos.x - half_size, end_pos.y + half_size));
					poly.push_back(ZQ_Vec2D(start_pos.x - half_size, start_pos.y + half_size));
					poly.push_back(ZQ_Vec2D(start_pos.x - half_size, start_pos.y - half_size));
					poly.push_back(ZQ_Vec2D(start_pos.x + half_size, start_pos.y - half_size));
				}
				else
				{
					poly.push_back(ZQ_Vec2D(end_pos.x + half_size, end_pos.y + half_size));
					poly.push_back(ZQ_Vec2D(end_pos.x - half_size, end_pos.y + half_size));
					poly.push_back(ZQ_Vec2D(end_pos.x - half_size, end_pos.y - half_size));
					poly.push_back(ZQ_Vec2D(start_pos.x - half_size, start_pos.y - half_size));
					poly.push_back(ZQ_Vec2D(start_pos.x + half_size, start_pos.y - half_size));
					poly.push_back(ZQ_Vec2D(start_pos.x + half_size, start_pos.y + half_size));
				}
			}
			else
			{
				if (end_pos.x >= start_pos.x)
				{
					poly.push_back(ZQ_Vec2D(end_pos.x - half_size, end_pos.y - half_size));
					poly.push_back(ZQ_Vec2D(end_pos.x + half_size, end_pos.y - half_size));
					poly.push_back(ZQ_Vec2D(end_pos.x + half_size, end_pos.y + half_size));
					poly.push_back(ZQ_Vec2D(start_pos.x + half_size, start_pos.y + half_size));
					poly.push_back(ZQ_Vec2D(start_pos.x - half_size, start_pos.y + half_size));
					poly.push_back(ZQ_Vec2D(start_pos.x - half_size, start_pos.y - half_size));
				}
				else
				{
					poly.push_back(ZQ_Vec2D(end_pos.x - half_size, end_pos.y + half_size));
					poly.push_back(ZQ_Vec2D(end_pos.x - half_size, end_pos.y - half_size));
					poly.push_back(ZQ_Vec2D(end_pos.x + half_size, end_pos.y - half_size));
					poly.push_back(ZQ_Vec2D(start_pos.x + half_size, start_pos.y - half_size));
					poly.push_back(ZQ_Vec2D(start_pos.x + half_size, start_pos.y + half_size));
					poly.push_back(ZQ_Vec2D(start_pos.x - half_size, start_pos.y + half_size));
				}
			}

			ZQ_ScanLinePolygonFill::ScanLinePolygonFillWithClip(poly, width, height, pixels);
			return true;
		}

		static bool ClipPolygon(const std::vector<ZQ_Vec2D>& in_poly, int width, int height, std::vector<ZQ_Vec2D>& out_poly)
		{
			std::vector<ZQ_Vec2D> tmp_poly;
			if (!_Sutherland_Hodgeman_clip_boundary(in_poly, tmp_poly, CLIPMODE_BOTTOM, width, height))
				return false;
			if (!_Sutherland_Hodgeman_clip_boundary(tmp_poly, out_poly, CLIPMODE_RIGHT, width, height))
				return false;
			if (!_Sutherland_Hodgeman_clip_boundary(out_poly, tmp_poly, CLIPMODE_TOP, width, height))
				return false;
			if (!_Sutherland_Hodgeman_clip_boundary(tmp_poly, out_poly, CLIPMODE_LEFT, width, height))
				return false;
			return true;
		}

	private:
		static void _getPolygonMinMax(const std::vector<ZQ_Vec2D>& polygon_pts, int& ymin, int& ymax)
		{
			int pts_num = polygon_pts.size();
			double fymin, fymax;
			fymin = polygon_pts[0].y;
			fymax = polygon_pts[0].y;
			for (int i = 1; i < pts_num; i++)
			{
				double cur_y = polygon_pts[i].y;
				if (fymax < cur_y)
					fymax = cur_y;
				if (fymin > cur_y)
					fymin = cur_y;
			}
			ymin = ceil(fymin);
			ymax = floor(fymax);
		}

		static void _initScanLineNewEdgeTable(const std::vector<ZQ_Vec2D>& polygon_pts, int ymin, int ymax, std::vector<std::vector<ScanLineEdge>>& slNet)
		{
			int pts_num = polygon_pts.size();
			ScanLineEdge e;
			for (int i = 0; i < pts_num; i++)
			{
				ZQ_Vec2D ps = polygon_pts[i];
				ZQ_Vec2D pe = polygon_pts[(i + 1) % pts_num];
				ZQ_Vec2D pss = polygon_pts[(i - 1 + pts_num) % pts_num];
				ZQ_Vec2D pee = polygon_pts[(i + 2) % pts_num];

				double floor_pe_y = floor(pe.y);
				bool ignore_parallel_case = (floor_pe_y < pe.y && floor_pe_y + 1 > pe.y && floor_pe_y < ps.y && floor_pe_y + 1 > ps.y) || (ps.y == pe.y);

				if (!ignore_parallel_case)
				{
					e.dx = (pe.x - ps.x) / (pe.y - ps.y);
					if (pe.y > ps.y)
					{
						int cur_y = ceil(ps.y) - ymin;
						e.xi = ps.x + (ceil(ps.y) - ps.y)*e.dx;
						if (pee.y >= pe.y)
							e.ymax = ceil(pe.y - 1);
						else
							e.ymax = floor(pe.y);
						if (e.ymax >= cur_y + ymin)
							slNet[cur_y].push_back(e);
					}
					else
					{
						int cur_y = ceil(pe.y) - ymin;
						e.xi = pe.x + (ceil(pe.y) - pe.y)*e.dx;
						if (pss.y >= ps.y)
							e.ymax = ceil(ps.y - 1);
						else
							e.ymax = floor(ps.y);
						if (e.ymax >= cur_y + ymin)
							slNet[cur_y].push_back(e);
					}
				}
			}
		}


		static void _processScanLineFill(const std::vector<std::vector<ScanLineEdge>>& slNet, int ymin, int ymax, std::vector<ZQ_Vec2D>& pixels)
		{
			std::vector<ScanLineEdge> aet;
			for (int y = ymin; y <= ymax; y++)
			{
				_insertNetListToAet(slNet[y - ymin], aet);
				_fillAetScanLine(aet, y, pixels);
				_removeNonActiveEdgeFromAet(aet, y);
				_updateAndResortAet(aet);
			}
		}

		static void _insertNetListToAet(const std::vector<ScanLineEdge>& net, std::vector<ScanLineEdge>& aet)
		{
			//insert sort
			for (int i = 0; i < net.size(); i++)
			{
				int aet_num = aet.size();
				aet.push_back(net[i]);

				if (aet_num == 0)
					continue;

				double xi = net[i].xi;
				int j = 0;
				for (; j < aet_num; j++)
				{
					if (xi < aet[j].xi)
						break;
				}
				if (j < aet_num)
				{
					for (int k = aet_num; k > j; k--)
						aet[k] = aet[k - 1];
					aet[j] = net[i];
				}
			}
		}

		static void _fillAetScanLine(const std::vector<ScanLineEdge>& aet, int y, std::vector<ZQ_Vec2D>& pixels)
		{
			int size = aet.size();
			if (size % 2 != 0)
			{
				printf("y = %d, odd cross\n", y);
			}
			for (int i = 0; i < size / 2; i++)
			{
				//[a,b)
				for (int x = ceil(aet[i * 2].xi); x < ceil(aet[i * 2 + 1].xi); x++)
					pixels.push_back(ZQ_Vec2D(x, y));
			}

		}

		static bool _isEdgeOutOfActive(const ScanLineEdge& e, int y)
		{
			return (e.ymax <= y);
		}

		static void _removeNonActiveEdgeFromAet(std::vector<ScanLineEdge>& aet, int y)
		{
			int size = aet.size();
			for (int i = size - 1; i >= 0; i--)
			{
				if (_isEdgeOutOfActive(aet[i], y))
					aet.erase(aet.begin() + i);
			}
		}

		static void _updateAetEdgeInfo(ScanLineEdge& e)
		{
			e.xi += e.dx;
		}


		static void _updateAndResortAet(std::vector<ScanLineEdge>& aet)
		{
			int size = aet.size();
			for (int i = 0; i < size; i++)
				_updateAetEdgeInfo(aet[i]);

			_sortAet(aet);

		}

		static void _sortAet(std::vector<ScanLineEdge>& aet)
		{
			int size = aet.size();
			for (int pass = 1; pass < size; pass++)
			{
				for (int i = 0; i < size - 1; i++)
				{
					if (aet[i].xi > aet[i + 1].xi)
					{
						ScanLineEdge e = aet[i];
						aet[i] = aet[i + 1];
						aet[i + 1] = e;
					}
				}
			}
		}

		static bool _Sutherland_Hodgeman_clip_boundary(const std::vector<ZQ_Vec2D>& in_poly, std::vector<ZQ_Vec2D>& out_poly, ClipMode mode, int width, int height)
		{
			out_poly.clear();
			int in_len = in_poly.size();
			if (in_len < 2)
				return true;
			switch (mode)
			{
			case CLIPMODE_LEFT: case CLIPMODE_RIGHT:case CLIPMODE_BOTTOM:case CLIPMODE_TOP:
				ZQ_Vec2D s = in_poly[in_len - 1];
				for (int j = 0; j < in_len; j++)
				{
					ZQ_Vec2D p = in_poly[j];
					if (_is_inside_clip_boundary(p, mode, width, height))
					{
						if (_is_inside_clip_boundary(s, mode, width, height))
						{
							out_poly.push_back(p);
						}
						else
						{
							ZQ_Vec2D cross_pt;
							if (!_intersect_with_clip_boundary(s, p, mode, cross_pt, width, height))
								return false;
							out_poly.push_back(cross_pt);
							out_poly.push_back(p);
						}
					}
					else if (_is_inside_clip_boundary(s, mode, width, height))
					{
						ZQ_Vec2D cross_pt;
						if (!_intersect_with_clip_boundary(s, p, mode, cross_pt, width, height))
							return false;
						out_poly.push_back(cross_pt);
					}
					s = p;
				}
				break;
			}
			return true;
		}


		static bool _is_inside_clip_boundary(const ZQ_Vec2D& p, ClipMode mode, int width, int height)
		{
			switch (mode)
			{
			case CLIPMODE_LEFT:
				return p.x >= 0;
				break;
			case CLIPMODE_RIGHT:
				return p.x <= width - 1;
				break;
			case CLIPMODE_BOTTOM:
				return p.y >= 0;
				break;
			case CLIPMODE_TOP:
				return p.y <= height - 1;
				break;
			}
			return false;
		}

		static bool _intersect_with_clip_boundary(const ZQ_Vec2D& s, const ZQ_Vec2D& p, ClipMode mode, ZQ_Vec2D& cross_pt, int width, int height)
		{
			switch (mode)
			{
			case CLIPMODE_LEFT:
				if (p.x == s.x)
					return false;
				cross_pt.x = 0;
				cross_pt.y = s.y + (p.y - s.y)*(0 - s.x) / (p.x - s.x);
				return true;
				break;
			case CLIPMODE_RIGHT:
				if (p.x == s.x)
					return false;
				cross_pt.x = width - 1;
				cross_pt.y = s.y + (p.y - s.y)*(width - 1 - s.x) / (p.x - s.x);
				return true;
				break;
			case CLIPMODE_BOTTOM:
				if (p.y == s.y)
					return false;
				cross_pt.y = 0;
				cross_pt.x = s.x + (p.x - s.x)*(0 - s.y) / (p.y - s.y);
				return true;
				break;
			case CLIPMODE_TOP:
				if (p.y == s.y)
					return false;
				cross_pt.y = height - 1;
				cross_pt.x = s.x + (p.x - s.x)*(height - 1 - s.y) / (p.y - s.y);
				return true;
				break;
			}
			return false;
		}
	};
}

#endif