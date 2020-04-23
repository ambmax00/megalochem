#ifndef ADC_ADC_RI_U1_H
#define ADC_ADC_RI_U1_H

#include "adc/adc_ops.h"

using namespace dbcsr;

namespace adc {
	
class adc0_u1 {
protected:
	
	vec<int> m_o;
	vec<int> m_v;
	
	svector<double> m_eps_o;
	svector<double> m_eps_v;
	
	stensor<2> m_f_oo;
	stensor<2> m_f_vv;
	
	stensor<2> m_sig_0;
	
	stensor<2> compute_zeroth_order(stensor<2>& u1) {
		
		// ADC0 : u_ia = - f_ij u_ja + f_ab u_ib
		//dbcsr::tensor<2> sig_0({.tensor_in = *u1, .name = "sig_0_1"});
		
		//dbcsr::einsum<2,2,2>({.x = "ij, ja -> ia", .t1 = *m_f_oo, .t2 = *u1, .t3 = sig_0, .alpha = -1.0});
		//dbcsr::einsum<2,2,2>({.x = "ab, ib -> ia", .t1 = *m_f_vv, .t2 = *u1, .t3 = sig_0, .beta = 1.0});
		
		tensor<2> sig_0 = tensor<2>::create_template().tensor_in(*u1).name("sig_0");
		
		dbcsr::contract(*m_f_oo, *u1, sig_0).alpha(-1.0).perform("ij, ja -> ia");
		dbcsr::contract(*m_f_vv, *u1, sig_0).beta(1.0).perform("ab, ib -> ia");
		
		return sig_0.get_stensor();
		
	}
	
public:

	adc0_u1(svector<double>& eps_o, svector<double>& eps_v, MPI_Comm comm, vec<int>& o, vec<int>& v) :
		m_eps_o(eps_o), m_eps_v(eps_v), m_o(o), m_v(v) {
			
		// set up f_oo and f_vv
		Eigen::VectorXd eigen_epso = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(m_eps_o->data(),m_eps_o->size());
		Eigen::VectorXd eigen_epsv = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(m_eps_v->data(),m_eps_v->size());

		Eigen::MatrixXd eigen_foo = eigen_epso.asDiagonal();
		Eigen::MatrixXd eigen_fvv = eigen_epsv.asDiagonal();
	
		dbcsr::pgrid<2> grid2(comm);
		
		vec<int> map1 = {0};
		vec<int> map2 = {1};
		arrvec<int,2> oo = {o,o};
		arrvec<int,2> vv = {v,v};
	
		auto _f_oo = dbcsr::eigen_to_tensor(eigen_foo, "f_oo", grid2, 
			map1, map2, oo);
		auto _f_vv = dbcsr::eigen_to_tensor(eigen_fvv, "f_vv", grid2, 
			map1, map2, vv);
	
		m_f_oo = _f_oo.get_stensor();
		m_f_vv = _f_vv.get_stensor();
		
		grid2.destroy();
		
	}
	
	virtual stensor<2> compute(stensor<2>& u1, double omega = 0.0) {
		
		return compute_zeroth_order(u1);
		
	}
		
	~adc0_u1() {}
	
}; // end class adc0_u1
		
		
	
class ri_adc1_u1 : public adc0_u1 {
protected:

	stensor<3> m_b_xoo;
	stensor<3> m_b_xov;
	stensor<3> m_b_xvv;
	
	// intermediates
	stensor<2> m_c_x;
	stensor<3> m_c_xov;
	
	vec<int> m_x;
	
	void compute_cx(stensor<2>& u1) {
		
		// c_X = b_Xia * u_ia
		std::cout << "Computing cx" << std::endl;
		
		vec<int> d = {1};
		
		if (!m_c_x) {
			dbcsr::pgrid<2> grid2(u1->comm());
			arrvec<int,2> xd = {m_x, d};
			m_c_x = make_stensor<2>(tensor<2>::create()
				.name("c_x").ngrid(grid2).map1({0}).map2({1}).blk_sizes(xd));
			grid2.destroy();
		}
		
		auto u_ovd = dbcsr::add_dummy(*u1);	
			
		//dbcsr::einsum<3,3,2>({.x = "Xjb, jbD -> XD", .t1 = *m_b_xov, .t2 = u_ovd, .t3 = *m_c_x});
		dbcsr::contract(*m_b_xov, u_ovd, *m_c_x).perform("Xjb, jbD -> XD");
		
		u_ovd.destroy();
		
	}
	
	void compute_cxov(stensor<2>& u1) {
		
		std::cout << "computing cxov" << std::endl;
		
		// ==== (X|ab) r_jb = c_xja
		
		if (!m_c_xov) {
			m_c_xov = make_stensor<3>(tensor<3>::create_template()
				.tensor_in(*m_b_xov).name("c_xov"));
		}
		
		//dbcsr::einsum<3,2,3>({.x = "Xab, jb -> Xja", .t1 = *m_b_xvv, .t2 = *u1, .t3 = *m_c_xov});
		dbcsr::contract(*m_b_xvv, *u1, *m_c_xov).perform("Xab, jb -> Xja");
	}
	
	dbcsr::stensor<2> compute_first_order(stensor<2>& u1) {
		
		// ADC 1
		// u_ia = [2*(ia|jb) - (ij|ab)] * r_ia
		
		dbcsr::pgrid<3> grid3(u1->comm());
		vec<int> d = {1};
		arrvec<int,3> ovd = {m_o,m_v,d};
		
		// ==== sig_1 = 2 * (ia|X) * c_X
		tensor<3> sig_1_d = tensor<3>::create().name("sig_1_d")
			.ngrid(grid3).map1({0,1}).map2({2}).blk_sizes(ovd);
		//dbcsr::tensor<2> sig_1_2(*r_ov, "sig_1_2");
		
		compute_cx(u1);	
		
		//dbcsr::einsum<3,2,3>({.x = "Xia, XD -> iaD", .t1 = *m_b_xov, .t2 = *m_c_x, .t3 = sig_1_d, .alpha = 2.0});
		dbcsr::contract(*m_b_xov, *m_c_x, sig_1_d).alpha(2.0).perform("Xia, XD -> iaD");
			
		stensor<2> sig_1 = std::make_shared<tensor<2>>(
			dbcsr::remove_dummy<3>(sig_1_d, vec<int>{0}, vec<int>{1},"sig_1"));
			
		sig_1_d.destroy();
			
		// ==== sig_1 += - (X|ij) c_xja 
			
		compute_cxov(u1);
		//dbcsr::einsum<3,3,2>({.x = "Xij, Xja -> ia", .t1 = *m_b_xoo, .t2 = *m_c_xov, .t3 = sig_1, .alpha = -1.0, .beta = 1.0});
		dbcsr::contract(*m_b_xoo, *m_c_xov, *sig_1).alpha(-1.0).beta(1.0).perform("Xij, Xja -> ia");
		
		return sig_1;
		
	}
		

public:

	ri_adc1_u1(svector<double>& eps_o, svector<double>& eps_v, stensor<3>& B_xoo, 
		stensor<3>& B_xov, stensor<3>& B_xvv) : 
		adc0_u1(
			eps_o, eps_v, 
			B_xov->comm(), 
			B_xov->blk_sizes()[1], 
			B_xov->blk_sizes()[2]
		), 
		m_x(B_xov->blk_sizes()[0]),
		m_b_xoo(B_xoo), 
		m_b_xov(B_xov), 
		m_b_xvv(B_xvv) {}
		
	virtual stensor<2> compute(stensor<2>& u1, double omega = 0.0) {
			
			// ADC0
			// form - f_ii r_ia + f_aa r_ia
			vec<int> d = {1};
			
			dbcsr::pgrid<2> grid2(u1->comm());
			
			// ADC 0
			auto sig_0 = compute_zeroth_order(u1);
			
			// ADC 1
			auto sig_1 = compute_first_order(u1);
			
			//dbcsr::copy<2>({.t_in = *sig_0, .t_out = *sig_1, .sum = true, .move_data = true});
			dbcsr::copy(*sig_0, *sig_1).sum(true).move_data(true).perform();
			
			return sig_1;
			
	}
	
}; // end class ri_adc1_u1

class ri_adc2_diis_u1 : ri_adc1_u1 {
protected:

	stensor<4> m_t_ovov;
	
	// sigma vector parts
	stensor<2> m_sig_0;
	stensor<2> m_sig_1;
	stensor<2> m_sig_V;
	stensor<2> m_sig_O;
	stensor<2> m_sig_OV1;
	stensor<2> m_sig_OV2;
	stensor<2> m_sig_OVOV1;
	stensor<2> m_sig_OVOV2;
	
// static intermediates

	stensor<2> m_i_oo;
	stensor<2> m_i_vv;

// dynamic intermediates

	stensor<2> m_i_ov_1;
	stensor<2> m_i_ov_2;
	stensor<4> m_r_ovov;

public:

	ri_adc2_diis_u1(svector<double> eps_o, svector<double> eps_v, 
		stensor<3>& B_xoo, 
		stensor<3>& B_xov, 
		stensor<3>& B_xvv,
		stensor<4>& t_ovov) :
		ri_adc1_u1(eps_o, eps_v, B_xoo, B_xov, B_xvv),
		m_t_ovov(t_ovov) {}
	
	void compute_start_imeds() {
		// compute I_oo and I_vv intermediates. Only done once
		// Ioo_ij = 1/2 sum_ckd [t_ickd (jc|kd) + t_jckd (ic|kd)]
		// Ivv_ab = 1/2 sum_kcl [t_kalc (kb|lc) + t_kblc (ka|lc)]
		
		std::cout << "Imeds" << std::endl;
		
		// step 1: compute f_ovX = t_ovov * i_ovX
		tensor<3> f_xov = tensor<3>::create_template().tensor_in(*m_b_xov).name("f_xov");
		//dbcsr::einsum<4,3,3>({.x = "iajb, Xjb -> Xia", .t1 = *m_t_ovov, .t2 = *m_b_xov, .t3 = f_xov});
		dbcsr::contract(*m_t_ovov, *m_b_xov, f_xov).perform("iajb, Xjb -> Xia");
		
		// step 2: compute Ioo_ij = 1/2 b_xjc * f_xic + 1/2 b_xic * f_xjc
		dbcsr::pgrid<2> grid2(f_xov.comm());
		arrvec<int,2> oo = {m_o, m_o};
		tensor<2> i_oo = tensor<2>::create().name("i_oo").ngrid(grid2) 
				.map1({0}).map2({1}).blk_sizes(oo);
		
		//dbcsr::einsum<3,3,2>({.x = "Xjc, Xic -> ij", .t1 = *m_b_xov, .t2 = f_xov, .t3 = i_oo, .alpha = 0.5});
		//dbcsr::einsum<3,3,2>({.x = "Xic, Xjc -> ij", 
		//	.t1 = *m_b_xov, .t2 = f_xov, .t3 = i_oo, .alpha = 0.5, .beta = 1.0});
		dbcsr::contract(*m_b_xov, f_xov, i_oo).alpha(0.5).perform("Xjc, Xic -> ij");
		dbcsr::contract(*m_b_xov, f_xov, i_oo).alpha(0.5).beta(1.0).perform("Xic, Xjc -> ij");
		
		// step 3: compute I_vv_ab = 1/2 b_xkb * f_xka + 1/2 b_xka * f_xkb
		arrvec<int,2> vv = {m_v, m_v};
		tensor<2> i_vv = tensor<2>::create().name("i_vv").ngrid(grid2)
			.map1({0}).map2({1}).blk_sizes(vv);
			
		//dbcsr::einsum<3,3,2>({.x = "Xkb, Xka -> ab", .t1 = *m_b_xov, .t2 = f_xov, .t3 = i_vv, .alpha = 0.5});
		//dbcsr::einsum<3,3,2>({.x = "Xka, Xkb -> ab", .t1 = *m_b_xov, .t2 = f_xov, .t3 = i_vv, .alpha = 0.5, .beta = 1.0});
		dbcsr::contract(*m_b_xov, f_xov, i_vv).alpha(0.5).perform("Xkb, Xka -> ab");
		dbcsr::contract(*m_b_xov, f_xov, i_vv).alpha(0.5).beta(1.0).perform("Xka, Xkb -> ab");
		
		m_i_oo = i_oo.get_stensor();
		m_i_vv = i_vv.get_stensor();
		
		dbcsr::print(*m_i_oo);
		dbcsr::print(*m_i_vv);
		
		f_xov.destroy();
		
	}
	
	void compute_iov1(stensor<2>& u1) {
		
		// compute I(1)_ia = sum_jb [2(ia|jb) - (ib|ja)] u_jb
		
		std::cout << "Iov" << std::endl;
		
		if (!m_i_ov_1) {
			m_i_ov_1 = make_stensor<2>(
				tensor<2>::create_template().tensor_in(*u1).name("i_ov_1"));
		}
		
		// step 1 i_ia = 2*b_xia cx
		vec<int> d = {1};
		
		dbcsr::pgrid<3> grid3(m_i_ov_1->comm());
		arrvec<int,3> ovd = {m_o,m_v,d};
		tensor<3> i_ovd = tensor<3>::create().name("i_ovd").ngrid(grid3).map1({0,1}).map2({2}).blk_sizes(ovd);
		
		//dbcsr::einsum<3,2,3>({.x = "Xia, Xd -> iad", .t1 = *m_b_xov, .t2 = *m_c_x, .t3 = i_int, .alpha = 2.0});
		dbcsr::contract(*m_b_xov, *m_c_x, i_ovd).alpha(2.0).perform("Xia, Xd -> iad");
		
		auto i_ov_1 = dbcsr::remove_dummy(i_ovd, vec<int>{0}, vec<int>{1}, "i_ov_1");
		i_ovd.destroy();
		
		m_i_ov_1 = i_ov_1.get_stensor();
		
		// step 2: I_xij = b_xib u_jb
		tensor<3> i_xoo = tensor<3>::create_template().tensor_in(*m_b_xoo).name("i_xoo");
		
		//dbcsr::einsum<3,2,3>({.x = "Xib, jb -> Xij", .t1 = *m_b_xov, .t2 = *u1, .t3 = i_xij});
		dbcsr::contract(*m_b_xov, *u1, i_xoo).perform("Xib, jb -> Xij");
		
		// step3: i_ia -= i_xij b_xja
		//dbcsr::einsum<3,3,2>({.x = "Xij, Xja -> ia", .t1 = i_xij, .t2 = *m_b_xov, .t3 = *m_i_ov_1, 
		//	.alpha = -1.0, .beta = 1.0});
		dbcsr::contract(i_xoo, *m_b_xov, *m_i_ov_1).alpha(-1.0).beta(1.0).perform("Xij, Xja -> ia");
		
		i_xoo.destroy();
		
	}
	
	void compute_iov2(stensor<2>& u1) {
		
		std::cout << "iov2" << std::endl;
		
		// compute I(2)_ia = sum_jb t_iajb u_jb
		auto u1_d = dbcsr::add_dummy(*u1);
		tensor<3> i_ovd = tensor<3>::create_template()
			.tensor_in(u1_d).name("i_ovd");
		
		//dbcsr::einsum<4,3,3>({.x = "iajb, jbD -> iaD", .t1 = *m_t_ovov, .t2 = u1_d, .t3 = iov_d});
		dbcsr::contract(*m_t_ovov, u1_d, i_ovd).perform("iajb, jbD -> iaD");
		
		auto i_ov_2 = dbcsr::remove_dummy(i_ovd, vec<int>{0}, vec<int>{1}, "i_ov_2");
		i_ovd.destroy();
		
		m_i_ov_2 = i_ov_2.get_stensor();
		
	}
	
	void compute_r_ovov(stensor<2>& u1, double omega) {
		
		std::cout << "R ovov" << std::endl;
		
		// compute R_iajb = [2(bar(ia)|jb) - (bar(ja)|ib) + 2 (ia|bar(jb)) - (ja|bar(ib))]/(D_iajb + omega)
		
		if (!m_r_ovov) {
			m_r_ovov = make_stensor<4>(tensor<4>::create_template()
				.tensor_in(*m_t_ovov).name("r_ovov"));
		}
		
		// step 1: (bar(ia)|X) = c_xia - sum_j b_xij u_ja
		tensor<3> J_xov = tensor<3>::create_template().tensor_in(*m_b_xov).name("J_xov");
		
		dbcsr::copy(*m_c_xov, J_xov).move_data(true).perform();
		
		//dbcsr::einsum<3,2,3>({.x = "Xij, ja -> Xia", .t1 = *m_b_xoo, .t2 = *u1, .t3 = J_xov,
		//	.alpha = -1.0, .beta = 1.0});
		dbcsr::contract(*m_b_xoo, *u1, J_xov).alpha(-1.0).beta(1.0).perform("Xij, ja -> Xia");
		
		// step 2: R_iajb = (bar(ia)|jb) + (ia|bar(jb))
		
		//dbcsr::einsum<3,3,4>({.x = "Xia, Xjb -> iajb", .t1 = J_xov, .t2 = *m_b_xov, .t3 = *m_r_ovov});
		//dbcsr::einsum<3,3,4>({.x = "Xia, Xjb -> iajb", .t1 = *m_b_xov, .t2 = J_xov, .t3 = *m_r_ovov,
		//	.beta = 1.0});
		dbcsr::contract(J_xov, *m_b_xov, *m_r_ovov).perform("Xia, Xjb -> iajb");
		dbcsr::contract(*m_b_xov, J_xov, *m_r_ovov).beta(1.0).perform("Xia, Xjb -> iajb");
		
		dbcsr::print(*m_r_ovov);
		
		// R_iajb = 2*R_iajb - R_ibja	
		antisym<double>(*m_r_ovov, 2.0);
		
		// scale 
		vec<double> eo = *m_eps_o;
		vec<double> ev = *m_eps_v;
		
		std::for_each(eo.begin(), eo.end(), [&omega](double& d) {
			d += 0.25 * omega;
		});
		std::for_each(ev.begin(), ev.end(), [&omega](double& d) {
			d -= 0.25 * omega;
		});
		
		scale<double>(*m_r_ovov, eo, ev, 1.0, -1.0);
		
	}
		
	dbcsr::stensor<2> compute(dbcsr::stensor<2>& u1, double omega = 0.0) {
			
			vec<int> d  = {1};
			
			std::cout << "U1" << std::endl;
			dbcsr::print(*u1);
			
			// ADC 0
			m_sig_0 = compute_zeroth_order(u1);
			
			// ADC 1
			m_sig_1 = compute_first_order(u1);
			
			// ADC 2 intermediates
			
			if (!m_i_oo && !m_i_vv) compute_start_imeds();
			
			compute_iov1(u1);
			compute_iov2(u1);
			
			dbcsr::print(*m_i_ov_1);
			dbcsr::print(*m_i_ov_2);
			
			std::cout << "E7" << std::endl;
			// sig_V_ia = i_ab u_ib
			if (!m_sig_V) m_sig_V = make_stensor<2>(
				tensor<2>::create_template().tensor_in(*u1).name("sig_V"));
			//dbcsr::einsum<2,2,2>({.x = "ab, ib -> ia", .t1 = *m_i_vv, .t2 = *u1, .t3 = *m_sig_V});
			dbcsr::contract(*m_i_vv, *u1, *m_sig_V).perform("ab, ib -> ia");
			
			std::cout << "E8" << std::endl;
			// sig_O_ia = i_ij u_ja
			if (!m_sig_O) m_sig_O = make_stensor<2>(
				tensor<2>::create_template().tensor_in(*u1).name("sig_O"));
			//dbcsr::einsum<2,2,2>({.x = "ij, ja -> ia", .t1 = *m_i_oo, .t2 = *u1, .t3 = *m_sig_O});
			dbcsr::contract(*m_i_oo, *u1, *m_sig_O).perform("ij, ja -> ia");
			
			std::cout << "E9" << std::endl;
			// sig_OV1_ia = - 0.5 * sum_jb t_iajb i_ov_jb
			dbcsr::pgrid<3> grid3(u1->comm());
			arrvec<int, 3> ovd = {m_o,m_v,d};
			tensor<3> i_ovd = tensor<3>::create().name("i_ovd").ngrid(grid3)
				.map1({0,1}).map2({2}).blk_sizes(ovd);
			std::cout << "E92" << std::endl;
			tensor<3> s_ovd = dbcsr::add_dummy(*m_sig_OV1);
			std::cout << "E93" << std::endl;
			//dbcsr::einsum<4,3,3>({.x = "iajb, jbD -> iaD", .t1 = *m_t_ovov, .t2 = i_ovd, .t3 = s_ovd, .alpha = -0.5});
			dbcsr::contract(*m_t_ovov, i_ovd, s_ovd).alpha(-0.5).perform("iajb, jbD -> iaD");
			std::cout << "E94" << std::endl;
			i_ovd.destroy();
			std::cout << "E95" << std::endl;
			m_sig_OV1 = std::make_shared<tensor<2>>(
				dbcsr::remove_dummy(s_ovd, vec<int>{0}, vec<int>{1},"sig_OV1"));
			
			std::cout << "E10" << std::endl;
			// sig_OV2_ia =  -0.5 * sum_jb [2(ia|jb) - (ib|ja)] i_ov2_jb
			if (!m_sig_OV2) m_sig_OV2 = make_stensor<2>(
				tensor<2>::create_template().tensor_in(*u1).name("sig_OV2"));
			
			std::cout << "E11" << std::endl;
			//step 1:
			// cXd = b_xjb i_ov2_jb
			dbcsr::pgrid<2> grid2(u1->comm());
			arrvec<int,2> xd = {m_x,d};
			tensor<2> cXD = tensor<2>::create().name("cXd").ngrid(grid2).map1({0}).map2({1}).blk_sizes(xd);
			auto i_ovd2 = dbcsr::add_dummy(*m_i_ov_2);
			//dbcsr::einsum<3,3,2>({.x = "Xjb, jbD -> XD", .t1 = *m_b_xov, .t2 = i_ovd2, .t3 = cXD});
			dbcsr::contract(*m_b_xov, i_ovd2, cXD).perform("Xjb, jbD -> XD");
			
			i_ovd2.destroy();
			
			std::cout << "E12" << std::endl;
			//step 2:
			// sig_OV2 = - 1 * b_xia * c_X
			
			tensor<3> sig_OV2_d = tensor<3>::create().name("sig_OV2_d").ngrid(grid3).map1({0,1}) 
				.map2({2}).blk_sizes(ovd);
				
			//dbcsr::einsum<3,2,3>({.x = "Xia, XD, iaD", .t1 = *m_b_xov, .t2 = cXD, .t3 = sig_OV2_d, .alpha = -1.0});
			dbcsr::contract(*m_b_xov, cXD, sig_OV2_d).alpha(-1.0).perform("Xia, XD, iaD");
			auto sig_OV2 = dbcsr::remove_dummy(sig_OV2_d, vec<int>{0}, vec<int>{1},"sig_OV2");
			
			sig_OV2_d.destroy();
			
			std::cout << "E13" << std::endl;
			//step 3:
			// c_Xij = b_xib i_jb
			arrvec<int,3> xoo = {m_x,m_o,m_o};
			tensor<3> cxoo = tensor<3>::create().name("cxoo").ngrid(grid3).map1({0}).map2({1,2}).blk_sizes(xoo);
			//dbcsr::einsum<3,2,3>({.x = "Xib, jb -> Xij", .t1 = *m_b_xov, .t2 = *m_i_ov_2, .t3 = cxoo});
			dbcsr::contract(*m_b_xov, *m_i_ov_2, cxoo).perform("Xib, jb -> Xij");
			
			std::cout << "E14" << std::endl;
			// step 4:
			// sig_OV2_ia += 0.5 * c_xij b_xja
			//dbcsr::einsum<3,3,2>({.x = "Xij, Xja -> ia", .t1 = cxoo, .t2 = *m_b_xov, .t3 = sig_OV2, .alpha = 0.5, .beta = 1.0});
			dbcsr::contract(cxoo, *m_b_xov, sig_OV2).alpha(0.5).beta(1.0).perform("Xij, Xja -> ia");
			
			m_sig_OV2 = sig_OV2.get_stensor();
			
			// make R_iajb
			
			compute_r_ovov(u1,omega);
			
			dbcsr::print(*m_r_ovov);
			
			std::cout << "E15" << std::endl;
			// Y_xia = b_xjb R_iajb
			tensor<3> Y_xov = tensor<3>::create_template().tensor_in(*m_b_xov).name("Y_xov");
			//dbcsr::einsum<4,3,3>({.x = "iajb, Xjb -> Xia", .t1 = *m_r_ovov, .t2 = *m_b_xov, .t3 = Y_xov});
			dbcsr::contract(*m_r_ovov, *m_b_xov, Y_xov).perform("iajb, Xjb -> Xia");
			
			m_r_ovov->clear();
			
			std::cout << "E16" << std::endl;
			// sig_OVOV1_ia = b_xab Y_xib
			if (!m_sig_OVOV1) m_sig_OVOV1 = make_stensor<2>(
				tensor<2>::create_template().tensor_in(*u1).name("sig_OVOV1"));
			
			//dbcsr::einsum<3,3,2>({.x = "Xab, Xib -> ia", .t1 = *m_b_xvv, .t2 = Y_xov, .t3 = *m_sig_OVOV1});
			dbcsr::contract(*m_b_xvv, Y_xov, *m_sig_OVOV1).perform("Xab, Xib -> ia");
			
			std::cout << "E17" << std::endl;
			// sig_OVOV2_ia = b_xij Y_xja
			if (!m_sig_OVOV2) m_sig_OVOV2 = make_stensor<2>(
				tensor<2>::create_template().tensor_in(*u1).name("sig_OVOV2"));
				
			//dbcsr::einsum<3,3,2>({.x = "Xij, Xja -> ia", .t1 = *m_b_xoo, .t2 = Y_xov, .t3 = *m_sig_OVOV2,
			//	.alpha = -1.0});*m_b_xoo, .t2 = Y_xov, .t3 = *m_sig_OVOV2
			dbcsr::contract(*m_b_xoo, Y_xov, *m_sig_OVOV2).alpha(-1.0).perform("Xij, Xja -> ia");
				
			std::cout << "SIG 0" << std::endl;
			dbcsr::print(*m_sig_0);
			
			std::cout << "SIG_1" << std::endl;
			dbcsr::print(*m_sig_1);
			
			std::cout << "SIG O" << std::endl;
			dbcsr::print(*m_sig_O);
			
			std::cout << "SIG V" << std::endl;
			dbcsr::print(*m_sig_V);
			
			std::cout << "SIG OV 1" << std::endl;
			dbcsr::print(*m_sig_OV1);
			
			std::cout << "SIG OV 2" << std::endl;
			dbcsr::print(*m_sig_OV2);
			
			std::cout << "SIG OVOV 1" << std::endl;
			dbcsr::print(*m_sig_OVOV1);
			
			std::cout << "SIG OVOV 2" << std::endl;
			dbcsr::print(*m_sig_OVOV2);
			
			//dbcsr::copy<2>({.t_in = *m_sig_1, .t_out = *m_sig_0, .sum = true});
			//dbcsr::copy<2>({.t_in = *m_sig_V, .t_out = *m_sig_0, .sum = true});
			//dbcsr::copy<2>({.t_in = *m_sig_O, .t_out = *m_sig_0, .sum = true});
			
			dbcsr::copy(*m_sig_1, *m_sig_0).sum(true).perform();
			dbcsr::copy(*m_sig_V, *m_sig_0).sum(true).perform();
			dbcsr::copy(*m_sig_O, *m_sig_0).sum(true).perform();
			
			std::cout << "sig now" << std::endl;
			dbcsr::print(*m_sig_0);
			
			//dbcsr::copy<2>({.t_in = *m_sig_OV1, .t_out = *m_sig_0, .sum = true});
			dbcsr::copy(*m_sig_OV1, *m_sig_0).sum(true).perform();
			
			std::cout << "sig h1" << std::endl;
			dbcsr::print(*m_sig_0);
			
			//dbcsr::copy<2>({.t_in = *m_sig_OV2, .t_out = *m_sig_0, .sum = true});
			dbcsr::copy(*m_sig_OV2, *m_sig_0).sum(true).perform();
			
			std::cout << "sig h2" << std::endl;
			dbcsr::print(*m_sig_0);
			
			//dbcsr::copy<2>({.t_in = *m_sig_OVOV1, .t_out = *m_sig_0, .sum = true});
			//dbcsr::copy<2>({.t_in = *m_sig_OVOV2, .t_out = *m_sig_0, .sum = true});
			dbcsr::copy(*m_sig_OVOV1, *m_sig_0).sum(true).perform();
			dbcsr::copy(*m_sig_OVOV2, *m_sig_0).sum(true).perform();
			
			dbcsr::print(*m_sig_0);
			
			return m_sig_0;
			
			
		}
	
};

/*

class sos_ri_adc2_diis_u1 : ri_adc1_u1 {
protected:

	dbcsr::stensor<4> m_t_ovov;
	
	// sigma vector parts
	dbcsr::stensor<2> m_sig_0;
	dbcsr::stensor<2> m_sig_1;
	dbcsr::stensor<2> m_sig_V;
	dbcsr::stensor<2> m_sig_O;
	dbcsr::stensor<2> m_sig_OV1;
	dbcsr::stensor<2> m_sig_OV2;
	dbcsr::stensor<2> m_sig_OVOV1;
	dbcsr::stensor<2> m_sig_OVOV2;
	
// static intermediates

	dbcsr::stensor<2> m_i_oo;
	dbcsr::stensor<2> m_i_vv;

// dynamic intermediates

	dbcsr::stensor<2> m_i_ov_1;
	dbcsr::stensor<2> m_i_ov_2;
	dbcsr::stensor<4> m_r_ovov;

public:

	sos_ri_adc2_diis_u1(svector<double> eps_o, svector<double> eps_v, 
		dbcsr::stensor<3>& B_xoo, 
		dbcsr::stensor<3>& B_xov, 
		dbcsr::stensor<3>& B_xvv,
		dbcsr::stensor<4>& t_ovov) :
		ri_adc1_u1(eps_o, eps_v, B_xoo, B_xov, B_xvv),
		m_t_ovov(t_ovov) {}
	
	void compute_start_imeds() {
		// compute I_oo and I_vv intermediates. Only done once
		// Ioo_ij = 1/2 sum_ckd [t_ickd^SOS (jc|kd) + t_jckd^SOS (ic|kd)]
		// Ivv_ab = 1/2 sum_kcl [t_kalc^SOS (kb|lc) + t_kblc^SOS (ka|lc)]
		
		std::cout << "Imeds" << std::endl;
		
		// step 1: compute f_ovX = t_ovov * i_ovX
		dbcsr::tensor<3> f_xov({.tensor_in = *m_b_xov, .name = "f_xov"});
		dbcsr::einsum<4,3,3>({.x = "iajb, Xjb -> Xia", .t1 = *m_t_ovov, .t2 = *m_b_xov, .t3 = f_xov});
		
		// step 2: compute Ioo_ij = 1/2 b_xjc * f_xic + 1/2 b_xic * f_xjc
		dbcsr::pgrid<2> grid2({.comm = f_xov.comm()});
		dbcsr::tensor<2> i_oo({.name = "i_oo", .pgridN = grid2, 
				.map1 = {0}, .map2 = {1}, .blk_sizes = {m_o,m_o}});
		
		dbcsr::einsum<3,3,2>({.x = "Xjc, Xic -> ij", .t1 = *m_b_xov, .t2 = f_xov, .t3 = i_oo, .alpha = 0.5});
		dbcsr::einsum<3,3,2>({.x = "Xic, Xjc -> ij", 
			.t1 = *m_b_xov, .t2 = f_xov, .t3 = i_oo, .alpha = 0.5, .beta = 1.0});
			
		// step 3: compute I_vv_ab = 1/2 b_xkb * f_xka + 1/2 b_xka * f_xkb
		dbcsr::tensor<2> i_vv({.name = "i_vv", .pgridN = grid2,
			.map1 = {0}, .map2 = {1}, .blk_sizes = {m_v,m_v}});
			
		dbcsr::einsum<3,3,2>({.x = "Xkb, Xka -> ab", .t1 = *m_b_xov, .t2 = f_xov, .t3 = i_vv, .alpha = 0.5});
		dbcsr::einsum<3,3,2>({.x = "Xka, Xkb -> ab", .t1 = *m_b_xov, .t2 = f_xov, .t3 = i_vv, .alpha = 0.5, .beta = 1.0});
		
		m_i_oo = i_oo.get_stensor();
		m_i_vv = i_vv.get_stensor();
		
		dbcsr::print(*m_i_oo);
		dbcsr::print(*m_i_vv);
		
		f_xov.destroy();
		
	}
	
	void compute_iov1(dbcsr::stensor<2>& u1) {
		
		// compute I(1)_ia = sum_jb [2(ia|jb) - (ib|ja)] u_jb
		
		std::cout << "Iov" << std::endl;
		
		if (!m_i_ov_1) {
			m_i_ov_1 = dbcsr::make_stensor<2>({.tensor_in = *u1, .name = "i_ov_1"});
		}
		
		// step 1 i_ia = 2*b_xia cx
		vec<int> d = {1};
		
		dbcsr::pgrid<3> grid3({.comm = m_i_ov_1->comm()});
		
		dbcsr::tensor<3> i_int({.name = "i_ov_int", .pgridN = grid3, .map1 = {0,1}, .map2 = {2}, .blk_sizes = {m_o,m_v,d}});
		
		dbcsr::einsum<3,2,3>({.x = "Xia, Xd -> iad", .t1 = *m_b_xov, .t2 = *m_c_x, .t3 = i_int, .alpha = 2.0});
		
		auto i_ov_1 = dbcsr::remove_dummy(i_int, vec<int>{0}, vec<int>{1});
		i_int.destroy();
		
		m_i_ov_1 = i_ov_1.get_stensor();
		
		// step 2: I_xij = b_xib u_jb
		dbcsr::tensor<3> i_xij({.tensor_in = *m_b_xoo, .name = "i_xij"});
		
		dbcsr::einsum<3,2,3>({.x = "Xib, jb -> Xij", .t1 = *m_b_xov, .t2 = *u1, .t3 = i_xij});
		
		// step3: i_ia -= i_xij b_xja
		dbcsr::einsum<3,3,2>({.x = "Xij, Xja -> ia", .t1 = i_xij, .t2 = *m_b_xov, .t3 = *m_i_ov_1, 
			.alpha = -1.0, .beta = 1.0});
			
		i_xij.destroy();
		
	}
	
	void compute_iov2(dbcsr::stensor<2>& u1) {
		
		std::cout << "iov2" << std::endl;
		
		// compute I(2)_ia = sum_jb t_iajb^SOS u_jb
		auto u1_d = dbcsr::add_dummy(*u1);
		dbcsr::tensor<3> iov_d({.tensor_in = u1_d, .name = "iov_d"});
		
		dbcsr::einsum<4,3,3>({.x = "iajb, jbD -> iaD", .t1 = *m_t_ovov, .t2 = u1_d, .t3 = iov_d});
		
		auto i_ov_2 = dbcsr::remove_dummy(iov_d, vec<int>{0}, vec<int>{1});
		iov_d.destroy();
		
		m_i_ov_2 = i_ov_2.get_stensor();
		
	}
	
	void compute_r_ovov(dbcsr::stensor<2>& u1, double omega) {
		
		std::cout << "R ovov" << std::endl;
		
		// compute R_iajb = c_os_coup * [(bar(ia)|jb) - (bar(ja)|ib) + 2 (ia|bar(jb)) - (ja|bar(ib))]
		// /(D_iajb + omega)
		
		if (!m_r_ovov) {
			m_r_ovov = dbcsr::make_stensor<4>({.tensor_in = *m_t_ovov, .name = "r_ovov"});
		}
		
		// step 1: (bar(ia)|X) = c_xia - sum_j b_xij u_ja
		dbcsr::tensor<3> J_xov({.tensor_in = *m_b_xov, .name = "J_xov"});
		
		dbcsr::copy<3>({.t_in = *m_c_xov, .t_out = J_xov, .move_data = true});
		
		dbcsr::einsum<3,2,3>({.x = "Xij, ja -> Xia", .t1 = *m_b_xoo, .t2 = *u1, .t3 = J_xov,
			.alpha = -1.0, .beta = 1.0});
			
		// step 2: R_iajb = (bar(ia)|jb) + (ia|bar(jb))
		
		dbcsr::einsum<3,3,4>({.x = "Xia, Xjb -> iajb", .t1 = J_xov, .t2 = *m_b_xov, .t3 = *m_r_ovov});
		dbcsr::einsum<3,3,4>({.x = "Xia, Xjb -> iajb", .t1 = *m_b_xov, .t2 = J_xov, .t3 = *m_r_ovov,
			.beta = 1.0});
		
		dbcsr::print(*m_r_ovov);
		
		// R_iajb = 2*R_iajb - R_ibja	
		//antisym<double>(*m_r_ovov, 2.0);
		
		// scale 
		vec<double> eo = *m_eps_o;
		vec<double> ev = *m_eps_v;
		
		std::for_each(eo.begin(), eo.end(), [&omega](double& d) {
			d += 0.25 * omega;
		});
		std::for_each(ev.begin(), ev.end(), [&omega](double& d) {
			d -= 0.25 * omega;
		});
		
		scale<double>(*m_r_ovov, eo, ev, 1.0, -1.0);
		m_r_ovov->scale(1.17);
		
	}
		
	dbcsr::stensor<2> compute(dbcsr::stensor<2>& u1, double omega = 0.0) {
			
			vec<int> d  = {1};
			
			std::cout << "U1" << std::endl;
			dbcsr::print(*u1);
			
			// ADC 0
			m_sig_0 = compute_zeroth_order(u1);
			
			// ADC 1
			m_sig_1 = compute_first_order(u1);
			
			//dbcsr::copy<2>({.t_in = *m_sig_1, .t_out = *m_sig_0, .sum = true});
			
			//dbcsr::print(*m_sig_0);
			
			dbcsr::pgrid<2> grid2({.comm = u1->comm()});
			dbcsr::pgrid<3> grid3({.comm = u1->comm()});
			
			// SOS ADC 2 intermediates
			
			if (!m_i_oo && !m_i_vv) compute_start_imeds();
			
			compute_iov1(u1);
			compute_iov2(u1);
			
			dbcsr::print(*m_i_ov_1);
			dbcsr::print(*m_i_ov_2);
			
			std::cout << "E7" << std::endl;
			// sig_V_ia = i_ab u_ib
			if (!m_sig_V) m_sig_V = dbcsr::make_stensor<2>({.tensor_in = *u1, .name = "sig_V"});
			dbcsr::einsum<2,2,2>({.x = "ab, ib -> ia", .t1 = *m_i_vv, .t2 = *u1, .t3 = *m_sig_V});
			
			std::cout << "E8" << std::endl;
			// sig_O_ia = i_ij u_ja
			if (!m_sig_O) m_sig_O = dbcsr::make_stensor<2>({.tensor_in = *u1, .name = "sig_O"});
			dbcsr::einsum<2,2,2>({.x = "ij, ja -> ia", .t1 = *m_i_oo, .t2 = *u1, .t3 = *m_sig_O});
			
			std::cout << "E9" << std::endl;
			// sig_OV1_ia = - 0.5 * sum_jb t_iajb^SOS i_ov_jb
			if (!m_sig_OV1) m_sig_OV1 = dbcsr::make_stensor<2>({.tensor_in = *u1, .name = "sig_OV1"});
			std::cout << "E91" << std::endl;
			auto i_ovd = dbcsr::add_dummy(*m_i_ov_1);
			std::cout << "E92" << std::endl;
			auto s_ovd = dbcsr::add_dummy(*m_sig_OV1);
			std::cout << "E93" << std::endl;
			dbcsr::einsum<4,3,3>({.x = "iajb, jbD -> iaD", .t1 = *m_t_ovov, .t2 = i_ovd, .t3 = s_ovd, .alpha = -0.5});
			std::cout << "E94" << std::endl;
			i_ovd.destroy();
			std::cout << "E95" << std::endl;
			*m_sig_OV1 = dbcsr::remove_dummy(s_ovd, vec<int>{0}, vec<int>{1});
			
			std::cout << "E10" << std::endl;
			// sig_OV2_ia =  -0.5 * sum_jb [2(ia|jb) - (ib|ja)] i_ov2_jb
			if (!m_sig_OV2) m_sig_OV2 = dbcsr::make_stensor<2>({.tensor_in = *u1, .name = "sig_OV2"});
			
			std::cout << "E11" << std::endl;
			//step 1:
			// cXd = b_xjb i_ov2_jb
			dbcsr::tensor<2> cXD({.name = "cXd", .pgridN = grid2, .map1 = {0}, .map2 = {1}, .blk_sizes = {m_x,d}});
			auto i_ovd2 = dbcsr::add_dummy(*m_i_ov_2);
			dbcsr::einsum<3,3,2>({.x = "Xjb, jbD -> XD", .t1 = *m_b_xov, .t2 = i_ovd2, .t3 = cXD});
			i_ovd2.destroy();
			
			std::cout << "E12" << std::endl;
			//step 2:
			// sig_OV2 = - 1 * b_xia * c_X
			dbcsr::tensor<3> sig_OV2_d({.name = "sig_OV2_d", .pgridN = grid3, .map1 = {0,1}, 
				.map2 = {2}, .blk_sizes = {m_o,m_v,d}});
				
			dbcsr::einsum<3,2,3>({.x = "Xia, XD, iaD", .t1 = *m_b_xov, .t2 = cXD, .t3 = sig_OV2_d, .alpha = -1.0});
			auto sig_OV2 = dbcsr::remove_dummy(sig_OV2_d, vec<int>{0}, vec<int>{1});
			
			sig_OV2_d.destroy();
			
			std::cout << "E13" << std::endl;
			//step 3:
			// c_Xij = b_xib i_jb
			dbcsr::tensor<3> cxoo({.name = "cxoo", .pgridN = grid3, .map1 = {0}, .map2 = {1,2}, .blk_sizes = {m_x,m_o,m_o}});
			dbcsr::einsum<3,2,3>({.x = "Xib, jb -> Xij", .t1 = *m_b_xov, .t2 = *m_i_ov_2, .t3 = cxoo});
			
			std::cout << "E14" << std::endl;
			// step 4:
			// sig_OV2_ia += 0.5 * c_xij b_xja
			dbcsr::einsum<3,3,2>({.x = "Xij, Xja -> ia", .t1 = cxoo, .t2 = *m_b_xov, .t3 = sig_OV2, .alpha = 0.5, .beta = 1.0});
			
			m_sig_OV2 = sig_OV2.get_stensor();
			
			// make R_iajb
			
			compute_r_ovov(u1,omega);
			
			dbcsr::print(*m_r_ovov);
			
			std::cout << "E15" << std::endl;
			// Y_xia = b_xjb R_iajb
			dbcsr::tensor<3> Y_xov({.tensor_in = *m_b_xov, .name = "Y_xov"});
			dbcsr::einsum<4,3,3>({.x = "iajb, Xjb -> Xia", .t1 = *m_r_ovov, .t2 = *m_b_xov, .t3 = Y_xov});
			
			m_r_ovov->clear();
			
			std::cout << "E16" << std::endl;
			// sig_OVOV1_ia = b_xab Y_xib
			if (!m_sig_OVOV1) m_sig_OVOV1 = dbcsr::make_stensor<2>({.tensor_in = *u1, .name = "sig_OVOV1"});
			
			dbcsr::einsum<3,3,2>({.x = "Xab, Xib -> ia", .t1 = *m_b_xvv, .t2 = Y_xov, .t3 = *m_sig_OVOV1});
			
			std::cout << "E17" << std::endl;
			// sig_OVOV2_ia = b_xij Y_xja
			if (!m_sig_OVOV2) m_sig_OVOV2 = dbcsr::make_stensor<2>({.tensor_in = *u1, .name = "sig_OVOV2"});
			
			dbcsr::einsum<3,3,2>({.x = "Xij, Xja -> ia", .t1 = *m_b_xoo, .t2 = Y_xov, .t3 = *m_sig_OVOV2,
				.alpha = -1.0});
				
			m_sig_OVOV1->scale(1.17);
			m_sig_OVOV2->scale(1.17);
			
			std::cout << "SIG 0" << std::endl;
			dbcsr::print(*m_sig_0);
			
			std::cout << "SIG_1" << std::endl;
			dbcsr::print(*m_sig_1);
			
			std::cout << "SIG O" << std::endl;
			dbcsr::print(*m_sig_O);
			
			std::cout << "SIG V" << std::endl;
			dbcsr::print(*m_sig_V);
			
			std::cout << "SIG OV 1" << std::endl;
			dbcsr::print(*m_sig_OV1);
			
			std::cout << "SIG OV 2" << std::endl;
			dbcsr::print(*m_sig_OV2);
			
			std::cout << "SIG OVOV 1" << std::endl;
			dbcsr::print(*m_sig_OVOV1);
			
			std::cout << "SIG OVOV 2" << std::endl;
			dbcsr::print(*m_sig_OVOV2);
			
			dbcsr::copy<2>({.t_in = *m_sig_1, .t_out = *m_sig_0, .sum = true});
			
			dbcsr::copy<2>({.t_in = *m_sig_V, .t_out = *m_sig_0, .sum = true});
			
			dbcsr::copy<2>({.t_in = *m_sig_O, .t_out = *m_sig_0, .sum = true});
			
			std::cout << "sig now" << std::endl;
			dbcsr::print(*m_sig_0);
			
			
			dbcsr::copy<2>({.t_in = *m_sig_OV1, .t_out = *m_sig_0, .sum = true});
			
			std::cout << "sig h1" << std::endl;
			dbcsr::print(*m_sig_0);
			
			dbcsr::copy<2>({.t_in = *m_sig_OV2, .t_out = *m_sig_0, .sum = true});
			
			std::cout << "sig h2" << std::endl;
			dbcsr::print(*m_sig_0);
			
			dbcsr::copy<2>({.t_in = *m_sig_OVOV1, .t_out = *m_sig_0, .sum = true});
			dbcsr::copy<2>({.t_in = *m_sig_OVOV2, .t_out = *m_sig_0, .sum = true});
			
			dbcsr::print(*m_sig_0);
			
			return m_sig_0;
			
			
		}
	
};*/
	
} // end namespace

#endif
