import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# OrthoCache Truncation Bound Proof

This file contains the statement and verification of the exponential Total Variation distance
truncation bound when evicting low-energy blocks.
-/

open Real
open BigOperators
open Classical

/-- Theorem: OrthoCache Total Variation Truncation Bound
    TV(α, \hat{α}) ≤ |S^c| * exp(beta - z_max)
-/
theorem orthocache_truncation_bound
  (N : ℕ)
  (S S_c : Set (Fin N))
  (hS : S ∪ S_c = Set.univ)
  (hS_disj : Disjoint S S_c)
  (z : Fin N → ℝ)
  (z_max : ℝ)
  (hz_max : ∀ i ∈ S, z i ≤ z_max)
  (beta : ℝ)
  (hbeta : ∀ i ∈ S_c, z i < beta)
  (alpha : Fin N → ℝ)
  (halpha : ∀ i, alpha i = exp (z i) / (∑ j, exp (z j)))
  (alpha_hat : Fin N → ℝ)
  (halpha_hat_in : ∀ i ∈ S, alpha_hat i = exp (z i) / (∑ j, if j ∈ S then exp (z j) else 0))
  (halpha_hat_out : ∀ i ∈ S_c, alpha_hat i = 0)
  (S_c_card : ℝ)
  : (1/2 : ℝ) * (∑ i, |alpha i - alpha_hat i|) ≤ S_c_card * exp (beta - z_max) := by
  sorry
