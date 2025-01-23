import torch


class AdaroundLoss:
    @staticmethod
    def compute_loss(
            orig_output: torch.Tensor,
            quantized_output: torch.Tensor,
            alpha: torch.Tensor,
            zeta,
            gamma,
            beta,
            lma):
        recon_loss = (torch.norm(quantized_output -
                      orig_output, p="fro", dim=1) ** 2).mean()

        h_alpha = torch.clamp(torch.sigmoid(alpha) * (zeta - gamma) +
                              gamma, 0, 1)
        reg_term = torch.add(
            1, -(torch.add(2 * h_alpha, -1).abs()).pow(beta)).sum()
        reg_loss = reg_term * lma
        return recon_loss + reg_loss
