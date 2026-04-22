from tkinter import RIGHT

from manim import *
import numpy as np

class LubricationEqs(Scene):
    def construct(self):
        # Define the equations
        eq1 = MathTex(r"\frac{\partial p}{\partial x} = \mu \frac{\partial^2 u}{\partial y^2}")
        eq2 = MathTex(r"\frac{\partial p}{\partial y} = 0")
        eq3 = MathTex(r"\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0")

        # Arrange the equations vertically
        equations = VGroup(eq1, eq2, eq3).arrange(DOWN, buff=0.5)

        # Add the equations to the scene
        self.play(Write(equations))
        self.wait(2)


from manim import *

class ThinFilmDerivation(Scene):
    def construct(self):
        # --- STAGE 1: Navier-Stokes & Scaling ---
        explanation = Text("Start with incompressible Navier-Stokes equations", font_size=30, color=WHITE).to_edge(UP)
        self.play(Write(explanation))

        # Equations 4, 5, 6
        ns_eqs = VGroup(
            MathTex(r"\frac{\partial u}{\partial x} + \frac{\partial w}{\partial z} = 0"),
            MathTex(r"\rho \left( \frac{Du}{Dt} \right) = -\frac{\partial p}{\partial x} + \mu \frac{\partial^2 u}{\partial z^2}"),
            MathTex(r"\rho \left( \frac{Dw}{Dt} \right) = -\frac{\partial p}{\partial z}")
        ).arrange(DOWN, buff=0.5).scale(0.8)
        ns_eqs.to_corner(UL).shift(DOWN*0.5)
        self.play(Write(ns_eqs))
        self.wait(1)

        # Scaling explanation (text)
        scaling_note = Tex(r"Lubrication Scaling: $\varepsilon = \frac{H}{L} << 1$", font_size=24).next_to(ns_eqs, RIGHT, buff=1).set_color(BLUE)
        self.play(Write(scaling_note))
        self.wait(1)

        # --- STAGE 2: Reduced Equations (7 & 8) ---
        reduced_eqs = VGroup(
            MathTex(r"\frac{\partial p}{\partial z} = 0 \implies p = p(x, t)"),
            MathTex(r"\mu \frac{\partial^2 u}{\partial z^2} = \frac{\partial p}{\partial x}")
        ).arrange(DOWN, buff=0.5).move_to(ORIGIN)

        self.play(
            FadeOut(ns_eqs[0]),
            FadeOut(scaling_note),
            ReplacementTransform(ns_eqs[1], reduced_eqs[1]), # Momentum -> Reduced
            ReplacementTransform(ns_eqs[2], reduced_eqs[0])  # Vertical NS -> p=p(x)
        )
        self.wait(2)

        # --- STAGE 3: Velocity Profile (13) ---
        # No-slip at z=0, No-shear at z=h
        bc_text = Text("BCs: u(0)=0, ∂u/∂z(h)=0", font_size=24, color=YELLOW).to_edge(UP, buff=1.5)
        
        u_profile = MathTex(
            r"u(z) = \frac{1}{2\mu} \frac{\partial p}{\partial x} (z^2 - 2hz)"
        ).scale(1.1)

        self.play(Write(bc_text))
        self.play(ReplacementTransform(reduced_eqs, u_profile))
        self.wait(2)

        # --- STAGE 4: Volumetric Flux (15) ---
        flux_int = MathTex(r"q = \int_0^h u(z) dz")
        flux_res = MathTex(r"q = -\frac{h^3}{3\mu} \frac{\partial p}{\partial x}")

        self.play(u_profile.animate.to_edge(LEFT).scale(0.7), bc_text.animate.to_edge(LEFT).scale(0.7))
        flux_int.next_to(u_profile, DOWN, buff=1)
        self.play(Write(flux_int))
        self.wait(1)
        
        flux_res.move_to(ORIGIN).scale(1.2)
        self.play(ReplacementTransform(flux_int, flux_res))
        self.wait(2)

        # --- STAGE 5: Capillary Pressure & Final Equation ---
        # Equations 16, 17, 20
        self.play(FadeOut(u_profile), FadeOut(bc_text))
        
        pressure = MathTex(r"p = p_0 - \gamma \frac{\partial^2 h}{\partial x^2}").to_edge(UP, buff=1.5)
        
        final_pde = MathTex(
            r"\frac{\partial h}{\partial t} + \frac{\gamma}{3\mu} \frac{\partial}{\partial x} \left( h^3 \frac{\partial^3 h}{\partial x^3} \right) = 0",
            color=GREEN
        ).scale(1.2)

        self.play(Write(pressure))
        self.play(ReplacementTransform(flux_res, final_pde))
        self.wait(3)

# Define custom colors for aesthetic
COL_PAPER = "#f2e6d5"  # Creamy paper color
COL_PAINT = "#1a3f90"  # Ocean Blue
GREEN_BKG = "#c0d75c"



class WatercolorScaleVisualization(MovingCameraScene):
    def construct(self):
        
        # 1. THE TITLE (Dynamic Positioning)
        #title = Title("Physical Justification for Scaling: Watercolor Paint")
        # We combine the move and scale into ONE updater for efficiency
        #title.add_updater(lambda m: m.move_to(self.camera.frame.get_center() + self.camera.frame.get_height() * 0.45 * UP).set_width(self.camera.frame.get_width() * 0.8))
        
        #self.add(title)
        #self.play(Write(title))

        # ==========================================================
        # STAGE 1: GLOBAL VIEW (Length Scale L)
        # ==========================================================

        self.camera.background_color = GREEN_BKG

        paper = Rectangle(width=16, height=4, color=COL_PAPER, fill_opacity=1).shift(DOWN*2)
        paper.set_fill([COL_PAPER, "#F4ECD8", COL_PAPER], opacity=1)

        self.add(paper)
        
        # Create a half-ellipse
        # width=1 (Length L), height=0.02 (Height H)
        paint_shape = Arc(radius=5, start_angle=0, angle=PI)
        paint_shape.stretch_to_fit_height(0.2) # This makes it "oblong"
        paint_shape.set_fill(COL_PAINT, opacity=0.6)
        paint_shape.set_stroke(COL_PAINT, width=2)

        paint_film = paint_shape.move_to(ORIGIN).shift(UP*0.1195) # Shift up to sit on paper

        global_label = Text("Global View: Length Scale L ~ 1 cm", font_size=24)
        global_label.to_edge(LEFT, buff=0.5).shift(UP).set_color(COL_PAINT)

        # ==========================================================
        # STAGE 3: LUBRICATION VIEW (Scaled H)
        # ==========================================================

        edge_point = paint_film.get_right() 
        
        local_label = Text("Lubrication View: Height Scale H ~ 0.2 mm", font_size=24)
        local_label.set_width(0.5)
        local_label.move_to(edge_point + LEFT * 0.1).shift(UP*0.15).set_color(COL_PAINT)

        # 1. Define the points in space
        top_pt = np.array([edge_point[0]+0.05, 0.21195, 0])    # (x, y, z)
        bottom_pt = np.array([edge_point[0]+0.05, 0.01195, 0])


        h_arrow = DoubleArrow(
            start=top_pt, 
            end=bottom_pt, 
            buff=0,
            tip_length=0.025, # Crucial for high zoom!
            stroke_width=0.5,
            color=COL_PAINT
        )

        label = MathTex("H").scale(0.15).next_to(h_arrow, RIGHT, buff=0.02).set_color(COL_PAINT)


        dashed_line = DashedLine(start=paint_film.get_left()+UP * 0.1 + LEFT * 4, end=paint_film.get_right()+UP * 0.1 + RIGHT * 4, color=COL_PAPER, dash_length=0.1, dashed_ratio=0.5, stroke_width=2)

        self.play(FadeIn(paint_film), Write(global_label))
        
        brace_l = Brace(paint_film, DOWN, color=COL_PAINT)
        text_l = brace_l.get_text("L").set_color(COL_PAINT) # Split color from method
        self.play(GrowFromCenter(brace_l), FadeIn(text_l), GrowFromCenter(dashed_line))
        self.wait(2)

        self.play(self.camera.frame.animate.scale(0.05).move_to(edge_point + LEFT * 0.1), FadeIn(local_label), FadeIn(h_arrow), FadeIn(label))

        # 2. ZOOM LOGIC
        # We clear the global UI before zooming
        self.play(FadeOut(global_label))
        
        self.wait(3)

        self.play(self.camera.frame.animate.scale(20).move_to(ORIGIN), FadeOut(local_label), FadeOut(h_arrow), FadeOut(label), FadeOut(global_label))

        self.wait(1)

        #
        #    Equations
        #

        eps_eq = MathTex(r"\varepsilon = \frac{H}{L} << 1").scale(0.75).move_to(ORIGIN + UP * 1).to_edge(LEFT, buff=0.5).set_color(COL_PAINT)

        assumption = Tex(r"Geometric Assumption:").scale(0.5).next_to(eps_eq, UP).set_color(COL_PAINT)

        bottom_frame = VGroup(
            eps_eq,
            assumption,
            paper,
            paint_film,
            dashed_line,
            brace_l,
            text_l
        )

        self.play(Write(eps_eq), Write(assumption))

        self.play(bottom_frame.animate.shift(DOWN * 2.5))

        self.wait(2)

        cons_momentum = Tex(r"Conservation of Momentum in 2D:").to_edge(UP, buff=0.25).scale(0.75).set_color(COL_PAINT)

        momentum_eq1 = MathTex(r"\rho", r" \left(\frac{D \mathbf{u}}{D t} \right)", r" = -\nabla p + \mu \nabla^2 \mathbf{u}").next_to(cons_momentum, DOWN, buff=0.5).scale(0.75).set_color(COL_PAINT)
        self.play(Write(cons_momentum), Write(momentum_eq1))

        reynolds_note = Tex(r"*Ignore inertial terms due to low $\varepsilon \cdot Re$").next_to(momentum_eq1, DOWN, buff=0.5).scale(0.6).set_color(COL_PAINT)

        slash_arrow = Arrow(
        start=momentum_eq1[1].get_corner(DL), 
        end=momentum_eq1[1].get_corner(UR) + UR*0.1, 
        buff=0, 
        tip_length=0.1,
        stroke_width=2
        ).set_color(RED)

        zero = MathTex("0").move_to(momentum_eq1[1].get_corner(UR) + UR*0.2).set_color(RED).scale(0.5)

        self.play(Write(reynolds_note))

        self.wait(2)

        self.play(Create(slash_arrow))
        self.play(Write(zero))

        canceled = VGroup(slash_arrow, zero)

        momentum_eq2 = MathTex(r"0 =", r"-\nabla p + \mu \nabla^2 \mathbf{u}").move_to(momentum_eq1.get_center()).scale(0.75).set_color(COL_PAINT)

        self.wait(2)

        self.play(ReplacementTransform(momentum_eq1, momentum_eq2), FadeOut(canceled), FadeOut(reynolds_note))

        self.wait(2)

        momentum_eq3 = MathTex(r"0 = ", r"-\frac{\partial p}{\partial z} +", r" \mu \frac{\partial^2 w}{\partial x^2}",
                               r"\\ 0 =", r"-\frac{\partial p}{\partial x} + \mu \frac{\partial^2 u}{\partial z^2}").move_to(momentum_eq1.get_center()).scale(0.75).set_color(COL_PAINT)
        
        momentum_eq3_zeros = VGroup(momentum_eq3[0], momentum_eq3[3])

        momentum_eq3_nonzeros = VGroup(momentum_eq3[1], momentum_eq3[2], momentum_eq3[4])

        self.play(ReplacementTransform(momentum_eq2[0], momentum_eq3_zeros), ReplacementTransform(momentum_eq2[1], momentum_eq3_nonzeros))

        self.wait(2)

        slash_arrow2 = Arrow(
        start=momentum_eq3[2].get_corner(DL), 
        end=momentum_eq3[2].get_corner(UR), 
        buff=0, 
        tip_length=0.1,
        stroke_width=2
        ).set_color(RED)

        zero2 = MathTex("0").move_to(momentum_eq3[2].get_corner(UR) + UR*0.1).set_color(RED).scale(0.5)

        dimensionless_scale = Tex(r"*$\frac{\Delta p_z}{\Delta p_x}$ is of order $\varepsilon^2$").move_to(reynolds_note.get_center()).scale(0.6).set_color(COL_PAINT)

        self.play(Write(dimensionless_scale))

        self.wait(2)

        self.play(Create(slash_arrow2))
        self.play(Write(zero2))

        self.wait(2)

        canceled2 = VGroup(slash_arrow2, zero2)

        momentum_eq4 = MathTex(r"\frac{\partial p}{\partial z}& = 0",
                                 r"\\ \frac{\partial p}{\partial x}& = \mu \frac{\partial^2 u}{\partial z^2}").move_to(momentum_eq1.get_center()).scale(0.75).set_color(COL_PAINT)
        
        momentum_x = MathTex(r"\frac{\partial p}{\partial x} = \mu \frac{\partial^2 u}{\partial z^2}").next_to(cons_momentum, DOWN, buff=0.5).scale(0.75).set_color(COL_PAINT)

        self.play(ReplacementTransform(momentum_eq3, momentum_eq4), FadeOut(canceled2), FadeOut(dimensionless_scale))
        
        self.wait(5)

        momx_title = Text("X-Momentum Equation").move_to(cons_momentum.get_center()).set_color(COL_PAINT).scale(0.5)

        self.play(FadeOut(momentum_eq4[0]), ReplacementTransform(cons_momentum, momx_title), momentum_eq4[1].animate.next_to(momx_title, DOWN, buff=0.5))

        monx_integrated1 = MathTex(r"\int_0^z \int_0^z \mu \frac{\partial^2 u}{\partial z^2} dz dz = \int_0^z \int_0^z \frac{\partial p}{\partial x} dz dz").move_to(momentum_x.get_center()).scale(0.75).set_color(COL_PAINT)

        self.wait(2)

        self.play(ReplacementTransform(momentum_eq4[1], monx_integrated1))

        self.wait(2)

        monx_integrated2 = MathTex(r"u(z) = \frac{1}{2 \mu} \frac{\partial p}{\partial x} z^2 + C_1 z", r"+", r"C_2").move_to(momentum_x.get_center()).scale(0.75).set_color(COL_PAINT)

        self.play(ReplacementTransform(monx_integrated1, monx_integrated2))

        c2_text = Tex(r"*Assuming $u(0) = 0 \implies C_2 = 0$").move_to(reynolds_note.get_center()).scale(0.75).set_color(COL_PAINT)

        self.play(Write(c2_text))

        # cross out and fade out C2
        slash_c2 = Arrow(
            start=monx_integrated2[2].get_corner(DL) + DL*0.1, 
            end=monx_integrated2[2].get_corner(UR) + UR*0.1,
            buff=0.1,
            tip_length=0.1,
            stroke_width=2
        ).set_color(RED)

        zero_c2 = MathTex("0").move_to(monx_integrated2[2].get_corner(UR) + UR*0.12).set_color(RED).scale(0.5)
        self.play(Create(slash_c2), Write(zero_c2))

        self.wait(2)

        monx_integrated3 = MathTex(r"u(z) =", r"\frac{1}{2 \mu} \frac{\partial p}{\partial x} z^2 + C_1 z").move_to(momentum_x.get_center()).scale(0.75).set_color(COL_PAINT)

        self.play(FadeOut(monx_integrated2[1:3]), FadeOut(zero_c2), FadeOut(slash_c2), monx_integrated2[0].animate.move_to(monx_integrated3.get_center()))

        self.wait(2)

        c1_text = Tex(r"*Assuming $\frac{\partial u}{\partial z}(h) = 0$ (i.e., shear-free boundary at $z = h$) $\implies C_1 = -\frac{1}{2 \mu} \frac{\partial p}{\partial x} h$").move_to(reynolds_note.get_center()).scale(0.75).set_color(COL_PAINT)

        monx_integrated4 = MathTex(r"u(z) =", r"\frac{1}{2 \mu} \frac{\partial p}{\partial x} z^2 - \frac{1}{2 \mu} \frac{\partial p}{\partial x} h z ").move_to(momentum_x.get_center()).scale(0.75).set_color(COL_PAINT)

        self.play(FadeOut(c2_text), FadeIn(monx_integrated3))

        self.play(Write(c1_text), FadeOut(monx_integrated2[0]))

        self.wait(2)

        self.play(ReplacementTransform(monx_integrated3, monx_integrated4))

        self.wait(2)

        monx_integrated5 = MathTex(r"u(z) =", r"\frac{1}{2 \mu} \frac{\partial p}{\partial x} (z^2 - 2hz)").move_to(momentum_x.get_center()).scale(0.75).set_color(COL_PAINT)
        self.play(ReplacementTransform(monx_integrated4[0], monx_integrated5[0]), ReplacementTransform(monx_integrated4[1], monx_integrated5[1]))

        self.wait(3)

        flux_eq = MathTex(r"q = \int_0^h", r"u(z)", r"dz").next_to(monx_integrated5, DOWN, buff=0.25).scale(0.75).set_color(COL_PAINT)

        flux_text = Tex(r"*Volumetric flux is obtained by integrating the velocity profile over the height").next_to(flux_eq, DOWN, buff=0.25).scale(0.75).set_color(COL_PAINT)

        self.play(FadeOut(c1_text))

        self.play(Write(flux_text))

        self.play(Write(flux_eq))

        flux_expanded = MathTex(r"q = \int_0^h", r"\frac{1}{2 \mu} \frac{\partial p}{\partial x} (z^2 - 2hz)", r"dz").move_to(monx_integrated5.get_center()).scale(0.75).set_color(COL_PAINT)

        mom_flux = VGroup(monx_integrated5[1], flux_eq)

        self.play(Indicate(flux_eq[1], color=RED), Indicate(monx_integrated5[1], color=RED))

        x_flux_title = Text("X-Momentum Flux").move_to(momx_title.get_center()).set_color(COL_PAINT).scale(0.5)

        self.wait(1)

        self.play(ReplacementTransform(monx_integrated5[1], flux_expanded[1]), ReplacementTransform(flux_eq[0], flux_expanded[0]),
                  ReplacementTransform(flux_eq[2], flux_expanded[2]), FadeOut(flux_eq[1]), FadeOut(monx_integrated5[0]), ReplacementTransform(momx_title, x_flux_title))

        self.wait(2)

        mom_flux_solved = MathTex(r"q = -\frac{h^3}{3 \mu}", r" \frac{\partial p}{\partial x}").move_to(monx_integrated5.get_center()).scale(0.75).set_color(COL_PAINT)

        self.play(ReplacementTransform(flux_expanded, mom_flux_solved))

        self.wait(3)
        
        self.play(FadeOut(flux_text))

        flux_objects = VGroup(mom_flux_solved, x_flux_title)

        self.play(flux_objects.animate.shift(LEFT * 3))

        youngpressure_title = Text("Young-Laplace Equation").move_to(x_flux_title.get_center()).shift(RIGHT * 6 + DOWN*0.05).set_color(COL_PAINT).scale(0.5)

        pressure_ddx = MathTex(r"\frac{\partial p}{\partial x}", r" = -\gamma \frac{\partial^3 h}{\partial x^3}").next_to(youngpressure_title, DOWN, buff=0.5).scale(0.75).set_color(COL_PAINT)

        pressure_eq = MathTex(r"p = p_0 - \gamma", r"\kappa").move_to(pressure_ddx.get_center()).scale(0.75).set_color(COL_PAINT)

        curvature_note = Tex(r"*Curvature, $\kappa$, is approximated as $\frac{\partial^2 h}{\partial x^2}$", r"\\ for small slopes").next_to(pressure_eq, DOWN, buff=0.5).scale(0.6).set_color(COL_PAINT)

        self.play(Write(youngpressure_title), Write(pressure_eq))

        self.wait(2)
        
        self.play(Write(curvature_note))

        self.wait(3)

        pressure_eq2 = MathTex(r"p = p_0 - \gamma", r"\frac{\partial^2 h}{\partial x^2}").move_to(pressure_eq.get_center()).scale(0.75).set_color(COL_PAINT)

        self.play(ReplacementTransform(pressure_eq[1], pressure_eq2[1]), ReplacementTransform(pressure_eq[0], pressure_eq2[0]))

        self.wait(2)

        self.play(ReplacementTransform(pressure_eq2, pressure_ddx))

        self.wait(2)

        self.play(Indicate(pressure_ddx[0], color=RED), Indicate(mom_flux_solved[1], color=RED))

        self.wait(1)

        final_xmom = MathTex(r"q = \frac{\gamma}{3 \mu} h^3 \frac{\partial^3 h}{\partial x^3}").next_to(x_flux_title, DOWN, buff=0.5).scale(0.75).set_color(COL_PAINT)

        pressure_flux = VGroup(pressure_ddx, mom_flux_solved)

        final_xmom2 = final_xmom.copy().move_to(final_xmom.get_center() + RIGHT * 3)

        x_flux_title2 = x_flux_title.copy().move_to(x_flux_title.get_center() + RIGHT * 3)

        self.play(ReplacementTransform(pressure_flux, final_xmom2), ReplacementTransform(x_flux_title, x_flux_title2), ReplacementTransform(youngpressure_title, x_flux_title2), FadeOut(curvature_note))

        self.wait(3)

        cons_mass_title = Text("Conservation of Mass").move_to(cons_momentum.get_center()).shift(RIGHT*3).set_color(COL_PAINT).scale(0.5)

        cons_mass_eq = MathTex(r"\frac{\partial h}{\partial t} +", r"\frac{\partial q}{\partial x}", r"= -E(x, t)").next_to(cons_mass_title, DOWN, buff=0.5).scale(0.75).set_color(COL_PAINT)

        x_flux_group = VGroup(x_flux_title2, final_xmom2)
        
        self.play(x_flux_group.animate.shift(LEFT * 3), Write(cons_mass_eq), Write(cons_mass_title))

        self.wait(2)

        q_ddx = MathTex(r"\frac{\partial q}{\partial x} = \frac{\gamma}{3 \mu} \frac{\partial}{\partial x} \left( h^3 \frac{\partial^3 h}{\partial x^3} \right)").move_to(final_xmom2.get_center()).scale(0.75).set_color(COL_PAINT)

        self.play(ReplacementTransform(final_xmom2, q_ddx))

        self.wait(2)
        
        self.play(Indicate(cons_mass_eq[1], color=RED), Indicate(q_ddx, color=RED))

        self.wait(2)

        final_pde = MathTex(r"\frac{\partial h}{\partial t} + \frac{\gamma}{3 \mu} \frac{\partial}{\partial x} \left( h^3 \frac{\partial^3 h}{\partial x^3} \right) =",r" -E(x, t)").move_to(q_ddx.get_center()).shift(RIGHT * 3).scale(0.75).set_color(COL_PAINT)

        mass_and_flux = VGroup(cons_mass_eq, q_ddx)

        thin_film_title = Text("Thin Film Equation").move_to(cons_mass_title.get_center()).shift(LEFT * 3).set_color(COL_PAINT).scale(0.5)

        titles = VGroup(cons_mass_title, x_flux_title2)

        self.play(ReplacementTransform(mass_and_flux, final_pde), ReplacementTransform(titles, thin_film_title))

        self.wait(6)

        